//! File-backed credential store for `auth.json`, ported from the upstream
//! persistence semantics of `packages/ai/src/cli.ts` (`loadAuth`/`saveAuth`)
//! and the `packages/ai/test/oauth.ts` harness (`loadAuthStorage`/
//! `saveAuthStorage`), expressed through the app-owned
//! [`CredentialStore`] trait (upstream `packages/coding-agent/src/core/
//! auth-storage.ts` `AuthStorage` is the app-level reference for the
//! read-modify-write shape).
//!
//! Document format (M2d controller ruling): `{ "<providerId>": credential }`
//! — the upstream `Record<string, Credential>` — pretty-printed with two
//! spaces, byte-compatible with upstream `JSON.stringify(auth, null, 2)`
//! ([`serde_json::to_string_pretty`] uses the same 2-space indent, `": "`
//! separators, and `{}`/`[]` for empty containers).
//!
//! Semantics:
//! - A missing file is an empty document (upstream `loadAuth`'s
//!   `existsSync` guard).
//! - An unparsable file fails visibly so a later write cannot replace saved
//!   credentials with a fresh document.
//! - Every write rewrites the whole document (`saveAuth` serializes the full
//!   map; `AuthStorage.modify` writes `{ ...currentData, [provider]: next }`).
//! - `delete` always rewrites the document, even when the key was absent
//!   (upstream `AuthStorage.delete` writes the filtered map unconditionally).
//! - File hygiene follows the `oauth.ts` harness: the parent directory is
//!   created recursively with mode `0o700`; writes use a private temporary
//!   file and atomic replacement with mode `0o600` on Unix.
//!
//! Deviations from upstream, disclosed:
//! - Key order: the document is stored as a `BTreeMap`, so `list` reports
//!   provider ids sorted and rewrites sort the keys. Upstream `Map` iteration
//!   preserves file insertion order. This is the established port-wide
//!   sorted-JSON-key deviation (serde_json `Map` is `BTreeMap`-backed), and
//!   the T1 round-trip oracle pinned a `BTreeMap` rewrite.
//! - Reads never resolve `!command` key configs (the coding-agent
//!   `AuthStorage.read` layer; app-owned, lands with the pi-rust app).
//! - No cross-process file lock. Upstream `cli.ts` and the `oauth.ts`
//!   harness lock nothing; the coding-agent's `proper-lockfile` is app-level.
//!   Within one process, writes serialize across the whole document.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use tokio::sync::Mutex;

use super::credential_store::{CredentialStore, ModifyCallback};
use super::types::{AuthError, AuthOperationOptions, Credential, CredentialInfo};

/// The `auth.json` document (upstream `AuthStorageData`).
type AuthDocument = BTreeMap<String, Credential>;

/// File-backed credential store keyed by provider id (upstream `cli.ts`
/// `loadAuth`/`saveAuth` over one `auth.json` path).
pub struct FileCredentialStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl FileCredentialStore {
    /// A store over one `auth.json` path. Upstream callers pass either the
    /// cwd-relative `auth.json` (`cli.ts`) or `<agent dir>/auth.json`
    /// (coding-agent); the path is caller-owned here.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    /// The backing `auth.json` path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Acquire the document writer lock, racing the operation's
    /// cancellation token (same semantics as the in-memory store).
    async fn lock_document(
        &self,
        options: &AuthOperationOptions,
    ) -> Result<tokio::sync::MutexGuard<'_, ()>, AuthError> {
        options.check()?;
        let guard = tokio::select! {
            guard = self.lock.lock() => Ok(guard),
            _ = options.cancelled() => Err(AuthError::Cancelled),
        }?;
        options.check()?;
        Ok(guard)
    }

    /// A missing file is empty; malformed saved credentials are an error.
    fn load_document(&self) -> Result<AuthDocument, AuthError> {
        let content = match std::fs::read_to_string(&self.path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AuthDocument::default());
            }
            Err(error) => {
                return Err(AuthError::Storage(format!(
                    "could not read {}: {error}",
                    self.path.display()
                )));
            }
        };
        serde_json::from_str(&content).map_err(|error| {
            AuthError::Storage(format!(
                "invalid saved credentials in {}: {error}",
                self.path.display()
            ))
        })
    }

    /// Write a complete owner-only document beside the destination, then
    /// atomically replace it so readers never see a partial credential set.
    fn save_document(&self, document: &AuthDocument) -> Result<(), AuthError> {
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if !parent.exists() {
            create_dir_private(parent).map_err(|error| {
                AuthError::Storage(format!("could not create {}: {error}", parent.display()))
            })?;
        }
        let body = serde_json::to_string_pretty(document).map_err(|error| {
            AuthError::Storage(format!("could not serialize auth.json: {error}"))
        })?;
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
            AuthError::Storage(format!("could not create credential temp file: {error}"))
        })?;
        restrict_file_mode(file.path()).map_err(|error| {
            AuthError::Storage(format!(
                "could not restrict credential temp file for {}: {error}",
                self.path.display()
            ))
        })?;
        file.write_all(body.as_bytes())
            .and_then(|_| file.as_file().sync_all())
            .map_err(|error| {
                AuthError::Storage(format!("could not write {}: {error}", self.path.display()))
            })?;
        file.persist(&self.path).map_err(|error| {
            AuthError::Storage(format!(
                "could not replace {}: {}",
                self.path.display(),
                error.error
            ))
        })?;
        Ok(())
    }
}

/// `mkdirSync(dir, { recursive: true, mode: 0o700 })` on unix.
fn create_dir_private(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// Restrict the temporary credential file before it receives secrets.
fn restrict_file_mode(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path)?.permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(path, permissions)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

impl CredentialStore for FileCredentialStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: &'a AuthOperationOptions,
    ) -> BoxFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            options.check()?;
            Ok(self.load_document()?.get(provider_id).cloned())
        })
    }

    fn list<'a>(
        &'a self,
        options: &'a AuthOperationOptions,
    ) -> BoxFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(async move {
            options.check()?;
            Ok(self
                .load_document()?
                .into_iter()
                .map(|(provider_id, credential)| CredentialInfo {
                    provider_id,
                    r#type: credential.auth_type(),
                })
                .collect())
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: ModifyCallback,
        options: &'a AuthOperationOptions,
    ) -> BoxFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            let _guard = self.lock_document(options).await?;
            let document = self.load_document()?;
            let current = document.get(provider_id).cloned();
            let next = {
                let input = current.clone();
                tokio::select! {
                    result = f(input) => result?,
                    _ = options.cancelled() => return Err(AuthError::Cancelled),
                }
            };
            // Upstream re-checks the signal after the callback so an abort
            // during the callback never applies its write.
            options.check()?;
            if let Some(next) = &next {
                let mut document = document;
                document.insert(provider_id.to_string(), next.clone());
                self.save_document(&document)?;
            }
            Ok(next.or(current))
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: &'a AuthOperationOptions,
    ) -> BoxFuture<'a, Result<(), AuthError>> {
        Box::pin(async move {
            let _guard = self.lock_document(options).await?;
            let mut document = self.load_document()?;
            document.remove(provider_id);
            // Upstream `AuthStorage.delete` rewrites the filtered map
            // unconditionally, so deleting the last entry leaves `{}`.
            self.save_document(&document)?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::auth::types::{ApiKeyCredential, OAuthCredential};
    use tempfile::TempDir;

    fn no_options() -> AuthOperationOptions {
        AuthOperationOptions::default()
    }

    fn api_key(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_string()),
            env: None,
            extra: BTreeMap::new(),
        })
    }

    fn oauth(access: &str, refresh: &str, expires: i64) -> Credential {
        Credential::OAuth(OAuthCredential {
            access: access.to_string(),
            refresh: refresh.to_string(),
            expires,
            extra: BTreeMap::new(),
        })
    }

    fn store_in(dir: &TempDir) -> FileCredentialStore {
        FileCredentialStore::new(dir.path().join("auth.json"))
    }

    fn read_file(dir: &TempDir) -> String {
        std::fs::read_to_string(dir.path().join("auth.json")).unwrap()
    }

    #[test]
    fn missing_file_is_an_empty_document() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        assert_eq!(
            futures::executor::block_on(store.read("p1", &no_options())).unwrap(),
            None
        );
        assert_eq!(
            futures::executor::block_on(store.list(&no_options())).unwrap(),
            Vec::new()
        );
        assert!(!dir.path().join("auth.json").exists());
    }

    #[test]
    fn a_corrupt_file_fails_without_losing_saved_credentials() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("auth.json"), "{not json").unwrap();
        let store = store_in(&dir);
        assert!(futures::executor::block_on(store.read("p1", &no_options())).is_err());
        assert!(futures::executor::block_on(store.list(&no_options())).is_err());
        assert!(futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("new"))) })),
            &no_options(),
        ))
        .is_err());
        assert_eq!(read_file(&dir), "{not json");
    }

    #[test]
    fn a_shape_invalid_entry_is_an_error() {
        // The port validates entries against the Credential schema at parse
        // time (upstream cli.ts casts unchecked): an oauth entry missing
        // `expires` is corrupt like a JSON syntax error.
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"p1":{"type":"oauth","refresh":"r","access":"a"}}"#,
        )
        .unwrap();
        let store = store_in(&dir);
        assert!(futures::executor::block_on(store.read("p1", &no_options())).is_err());
    }

    #[test]
    fn modify_writes_the_upstream_pretty_document() {
        // `JSON.stringify(auth, null, 2)`: 2-space indent, `": "` separator.
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        futures::executor::block_on(store.modify(
            "anthropic",
            Box::new(|current| {
                assert_eq!(current, None);
                Box::pin(async { Ok(Some(oauth("a", "r", 10))) })
            }),
            &no_options(),
        ))
        .unwrap();
        assert_eq!(
            read_file(&dir),
            "{\n  \"anthropic\": {\n    \"type\": \"oauth\",\n    \"refresh\": \"r\",\n    \
             \"access\": \"a\",\n    \"expires\": 10\n  }\n}"
        );
    }

    #[test]
    fn the_whole_document_round_trips_upstream_bytes() {
        // An upstream-written (JS insertion order) document parses, and a
        // rewrite re-serializes entry-for-entry (keys sorted by the
        // disclosed BTreeMap deviation).
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(
            &path,
            concat!(
                "{\n",
                "  \"openai\": {\n",
                "    \"type\": \"api_key\",\n",
                "    \"key\": \"sk\"\n",
                "  },\n",
                "  \"anthropic\": {\n",
                "    \"type\": \"oauth\",\n",
                "    \"refresh\": \"r\",\n",
                "    \"access\": \"a\",\n",
                "    \"expires\": 1735689600000,\n",
                "    \"accountId\": \"acc\"\n",
                "  }\n",
                "}"
            ),
        )
        .unwrap();
        let store = FileCredentialStore::new(&path);
        assert_eq!(
            futures::executor::block_on(store.read("anthropic", &no_options())).unwrap(),
            Some(Credential::OAuth(OAuthCredential {
                refresh: "r".to_string(),
                access: "a".to_string(),
                expires: 1_735_689_600_000,
                extra: [(
                    "accountId".to_string(),
                    serde_json::Value::String("acc".to_string())
                )]
                .into_iter()
                .collect(),
            }))
        );
        // Rewrite through modify: same entries, sorted keys, unknown OAuth
        // extension field preserved.
        futures::executor::block_on(store.modify(
            "anthropic",
            Box::new(|current| Box::pin(async move { Ok(current) })),
            &no_options(),
        ))
        .unwrap();
        assert_eq!(
            read_file(&dir),
            concat!(
                "{\n",
                "  \"anthropic\": {\n",
                "    \"type\": \"oauth\",\n",
                "    \"refresh\": \"r\",\n",
                "    \"access\": \"a\",\n",
                "    \"expires\": 1735689600000,\n",
                "    \"accountId\": \"acc\"\n",
                "  },\n",
                "  \"openai\": {\n",
                "    \"type\": \"api_key\",\n",
                "    \"key\": \"sk\"\n",
                "  }\n",
                "}"
            )
        );
    }

    #[test]
    fn api_key_unknown_fields_survive_a_file_rewrite() {
        // The T1 ruling carried here: upstream JS preserves unknown fields on
        // api_key entries through loadAuth/saveAuth (JSON.parse keeps them);
        // the port's `extra` map must too.
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"cloudflare":{"type":"api_key","key":"k","env":{"ACCOUNT_ID":"abc"},"note":"kept"}}"#,
        )
        .unwrap();
        let store = store_in(&dir);
        futures::executor::block_on(store.modify(
            "cloudflare",
            Box::new(|current| Box::pin(async move { Ok(current) })),
            &no_options(),
        ))
        .unwrap();
        assert_eq!(
            read_file(&dir),
            "{\n  \"cloudflare\": {\n    \"type\": \"api_key\",\n    \"key\": \"k\",\n    \
             \"env\": {\n      \"ACCOUNT_ID\": \"abc\"\n    },\n    \"note\": \"kept\"\n  }\n}"
        );
    }

    #[test]
    fn list_reports_sorted_entries_without_secrets() {
        // Disclosed deviation: upstream Map iteration preserves file order;
        // the BTreeMap document reports provider ids sorted.
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"zzz":{"type":"api_key","key":"k"},"aaa":{"type":"oauth","refresh":"r","access":"a","expires":1}}"#,
        )
        .unwrap();
        let store = store_in(&dir);
        assert_eq!(
            futures::executor::block_on(store.list(&no_options())).unwrap(),
            vec![
                CredentialInfo {
                    provider_id: "aaa".to_string(),
                    r#type: crate::ai::auth::types::AuthType::OAuth,
                },
                CredentialInfo {
                    provider_id: "zzz".to_string(),
                    r#type: crate::ai::auth::types::AuthType::ApiKey,
                },
            ]
        );
    }

    #[test]
    fn modify_is_a_serialized_read_modify_write_of_the_whole_document() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("first"))) })),
            &no_options(),
        ))
        .unwrap();
        // The second write sees the first and keeps its entry.
        futures::executor::block_on(store.modify(
            "p2",
            Box::new(|current| {
                assert_eq!(current, None);
                Box::pin(async { Ok(Some(oauth("a", "r", 1))) })
            }),
            &no_options(),
        ))
        .unwrap();
        let document: AuthDocument = serde_json::from_str(&read_file(&dir)).unwrap();
        assert_eq!(document.len(), 2);
        assert_eq!(document.get("p1"), Some(&api_key("first")));

        // Returning None leaves the document untouched on disk.
        let before = read_file(&dir);
        futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(None) })),
            &no_options(),
        ))
        .unwrap();
        assert_eq!(read_file(&dir), before);
    }

    #[test]
    fn a_modify_callback_error_propagates_and_never_writes() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"p1":{"type":"api_key","key":"old"}}"#,
        )
        .unwrap();
        let store = store_in(&dir);
        let error = futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| {
                Box::pin(async { Err(AuthError::Operation("refresh failed".to_string())) })
            }),
            &no_options(),
        ))
        .unwrap_err();
        assert_eq!(error, AuthError::Operation("refresh failed".to_string()));
        // A failed callback never writes: the file keeps the raw bytes it
        // was seeded with (still the un-pretty single-line document).
        assert_eq!(read_file(&dir), r#"{"p1":{"type":"api_key","key":"old"}}"#);
    }

    #[test]
    fn a_cancelled_modify_never_writes() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let token = tokio_util::sync::CancellationToken::new();
        let options = AuthOperationOptions::new(token.clone());
        token.cancel();
        let error = futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("late"))) })),
            &options,
        ))
        .unwrap_err();
        assert_eq!(error, AuthError::Cancelled);
        assert!(!dir.path().join("auth.json").exists());
    }

    #[test]
    fn delete_removes_the_entry_and_rewrites_the_document() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"p1":{"type":"api_key","key":"k"},"p2":{"type":"api_key","key":"k2"}}"#,
        )
        .unwrap();
        let store = store_in(&dir);
        futures::executor::block_on(store.delete("p1", &no_options())).unwrap();
        assert_eq!(
            read_file(&dir),
            "{\n  \"p2\": {\n    \"type\": \"api_key\",\n    \"key\": \"k2\"\n  }\n}"
        );
        // Deleting the last entry leaves `{}`, like upstream's unconditional
        // filtered-map rewrite; deleting an absent entry is still fine.
        futures::executor::block_on(store.delete("p2", &no_options())).unwrap();
        assert_eq!(read_file(&dir), "{}");
        futures::executor::block_on(store.delete("absent", &no_options())).unwrap();
        assert_eq!(read_file(&dir), "{}");
    }

    #[tokio::test]
    async fn deletes_serialize_against_concurrent_modifies() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(store_in(&dir));
        let (release, wait) = tokio::sync::oneshot::channel::<()>();
        let (started, started_rx) = tokio::sync::oneshot::channel::<()>();
        let writer = store.clone();
        let write = tokio::spawn(async move {
            writer
                .modify(
                    "p1",
                    Box::new(move |_| {
                        Box::pin(async move {
                            started.send(()).ok();
                            wait.await.ok();
                            Ok(Some(api_key("written")))
                        })
                    }),
                    &no_options(),
                )
                .await
                .unwrap();
        });
        started_rx.await.unwrap();
        // The delete queues behind the in-flight modify (a yield lets it
        // reach the provider lock) and cannot lose the writer's entry.
        let deleter = store.clone();
        let delete = tokio::spawn(async move {
            deleter.delete("p1", &no_options()).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        release.send(()).ok();
        write.await.unwrap();
        delete.await.unwrap();
        // Final `{}` proves the ordering: a delete that had beaten the
        // modify would have removed nothing and the writer would have
        // re-written the entry afterwards.
        assert_eq!(read_file(&dir), "{}");
    }

    #[tokio::test]
    async fn different_provider_writes_do_not_overwrite_each_other() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(store_in(&dir));
        let (release, wait) = tokio::sync::oneshot::channel::<()>();
        let (started, started_rx) = tokio::sync::oneshot::channel::<()>();
        let first = store.clone();
        let write_first = tokio::spawn(async move {
            first
                .modify(
                    "openai",
                    Box::new(move |_| {
                        Box::pin(async move {
                            started.send(()).ok();
                            wait.await.ok();
                            Ok(Some(api_key("first")))
                        })
                    }),
                    &no_options(),
                )
                .await
                .unwrap();
        });
        started_rx.await.unwrap();
        let second = store.clone();
        let (entered, mut entered_rx) = tokio::sync::oneshot::channel::<()>();
        let write_second = tokio::spawn(async move {
            second
                .modify(
                    "anthropic",
                    Box::new(move |_| {
                        Box::pin(async move {
                            entered.send(()).ok();
                            Ok(Some(api_key("second")))
                        })
                    }),
                    &no_options(),
                )
                .await
                .unwrap();
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut entered_rx)
                .await
                .is_err()
        );
        release.send(()).ok();
        write_first.await.unwrap();
        write_second.await.unwrap();
        let document: AuthDocument = serde_json::from_str(&read_file(&dir)).unwrap();
        assert_eq!(document.get("openai"), Some(&api_key("first")));
        assert_eq!(document.get("anthropic"), Some(&api_key("second")));
    }

    #[tokio::test]
    async fn cancelled_queued_delete_does_not_remove_a_credential() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(store_in(&dir));
        let (release, wait) = tokio::sync::oneshot::channel::<()>();
        let (started, started_rx) = tokio::sync::oneshot::channel::<()>();
        let writer = store.clone();
        let write = tokio::spawn(async move {
            writer
                .modify(
                    "openai",
                    Box::new(move |_| {
                        Box::pin(async move {
                            started.send(()).ok();
                            wait.await.ok();
                            Ok(Some(api_key("kept")))
                        })
                    }),
                    &no_options(),
                )
                .await
                .unwrap();
        });
        started_rx.await.unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        let deleter = store.clone();
        let options = AuthOperationOptions::new(token.clone());
        let delete = tokio::spawn(async move { deleter.delete("openai", &options).await });
        token.cancel();
        release.send(()).ok();
        write.await.unwrap();
        assert_eq!(delete.await.unwrap(), Err(AuthError::Cancelled));
        assert_eq!(
            futures::executor::block_on(store.read("openai", &no_options())).unwrap(),
            Some(api_key("kept"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn saves_create_a_private_directory_and_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let store = FileCredentialStore::new(dir.path().join("nested").join("auth.json"));
        futures::executor::block_on(store.modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("k"))) })),
            &no_options(),
        ))
        .unwrap();
        let mode = std::fs::metadata(dir.path().join("nested").join("auth.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn replacement_restricts_an_existing_permissive_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(&path, "{}").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let store = FileCredentialStore::new(&path);
        futures::executor::block_on(store.modify(
            "openai",
            Box::new(|_| Box::pin(async { Ok(Some(api_key("private"))) })),
            &no_options(),
        ))
        .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
