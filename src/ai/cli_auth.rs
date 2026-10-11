//! Login/logout command functions ported from the interactive CLI of
//! upstream `packages/ai/src/cli.ts` (`login`, lines 45-77). The binary's
//! argument wiring (`login [provider]`, provider selection, help/list) is
//! the port's CLI layer (M2d T9); this module owns the storage-adjacent
//! behavior: run a provider's OAuth login flow, then persist the returned
//! credential under the provider id in `auth.json`.
//!
//! Mapping to upstream, per the M2d controller ruling that interactive
//! surfaces go through [`AuthInteraction`]:
//! - Upstream wires `prompt`/`notify` to readline/stdio inside `login`; the
//!   port receives a caller-built [`AuthInteraction`] (the console impl is
//!   the CLI's). A caller cancellation token is honored through token
//!   exchange and credential persistence.
//! - Upstream prints `\nCredentials saved to ${AUTH_FILE}` with
//!   `console.log`; the port notifies [`AuthEvent::Info`] with the same
//!   message text ("Credentials saved to auth.json" — upstream hardcodes the
//!   `AUTH_FILE` literal). The leading newline and rendering belong to the
//!   console interaction.
//!
//! Upstream `cli.ts` has no logout (deletion is the [`CredentialStore`]
//! `delete` contract, "Remove a credential (logout)"); [`logout`] is the
//! port's minimal command over it.

use std::sync::Arc;

use crate::ai::auth::credential_store::CredentialStore;
use crate::ai::auth::types::{
    AuthError, AuthEvent, AuthInteraction, AuthOperationOptions, Credential, OAuthAuth,
    OAuthCredential, ProviderAuthInteraction,
};

/// Upstream `login(providerId)` (cli.ts:45-77): run the provider's OAuth
/// login through the interaction, then store the credential — upstream
/// `auth[providerId] = credential; saveAuth(auth)`.
pub async fn login(
    oauth: &dyn OAuthAuth,
    provider_id: &str,
    store: &dyn CredentialStore,
    interaction: Arc<dyn AuthInteraction>,
    options: &AuthOperationOptions,
) -> Result<OAuthCredential, AuthError> {
    login_with_device_id(oauth, provider_id, store, interaction, None, options).await
}

/// The cli.ts delta's login shape: the app-supplied `LoginOptions`
/// (`{ getDeviceId: randomUUID }` upstream) ride on the normalized
/// interaction for flows that need the installation ID.
#[allow(clippy::too_many_arguments)]
pub async fn login_with_device_id(
    oauth: &dyn OAuthAuth,
    provider_id: &str,
    store: &dyn CredentialStore,
    interaction: Arc<dyn AuthInteraction>,
    login_options: Option<std::sync::Arc<crate::ai::auth::types::LoginOptions>>,
    options: &AuthOperationOptions,
) -> Result<OAuthCredential, AuthError> {
    let signal = options
        .signal
        .clone()
        .or_else(|| interaction.signal())
        .unwrap_or_default();
    if signal.is_cancelled() {
        return Err(AuthError::Cancelled);
    }
    let normalized = ProviderAuthInteraction::new(interaction.clone(), signal.clone())
        .with_login_options(login_options);
    let credential = oauth.login(normalized).await?;
    if signal.is_cancelled() {
        return Err(AuthError::Cancelled);
    }
    let stored = credential.clone();
    store
        .modify(
            provider_id,
            Box::new(move |_| Box::pin(async move { Ok(Some(Credential::OAuth(stored))) })),
            &AuthOperationOptions::new(signal),
        )
        .await?;
    // Upstream cli.ts:73 `console.log(\`\nCredentials saved to ${AUTH_FILE}\`)`;
    // the leading newline is the console surface's rendering.
    interaction.notify(AuthEvent::Info {
        message: "Credentials saved to auth.json".to_string(),
        links: None,
    });
    Ok(credential)
}

/// Remove the provider's stored credential (upstream `CredentialStore`
/// `delete`: "Remove a credential (logout)"). No upstream `cli.ts`
/// counterpart — success/failure messaging is the CLI surface's (T9).
pub async fn logout(
    store: &dyn CredentialStore,
    provider_id: &str,
    options: &AuthOperationOptions,
) -> Result<(), AuthError> {
    store.delete(provider_id, options).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::auth::credential_store::InMemoryCredentialStore as InMemoryStore;
    use crate::ai::auth::types::{ApiKeyCredential, AuthPrompt, ModelAuth};
    use futures::future::BoxFuture;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    fn no_options() -> AuthOperationOptions {
        AuthOperationOptions::default()
    }

    /// OAuth flow returning a canned credential, recording the interaction
    /// it was given (upstream `provider.auth.oauth.login`).
    struct FakeOAuth {
        credential: OAuthCredential,
        received_signals: Mutex<Vec<Option<CancellationToken>>>,
    }

    impl OAuthAuth for FakeOAuth {
        fn name(&self) -> &str {
            "Fake provider"
        }

        fn login<'a>(
            &'a self,
            interaction: ProviderAuthInteraction,
        ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
            self.received_signals
                .lock()
                .unwrap()
                .push(interaction.signal());
            Box::pin(async move { Ok(self.credential.clone()) })
        }

        fn refresh<'a>(
            &'a self,
            _credential: OAuthCredential,
            _options: &'a AuthOperationOptions,
        ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
            unreachable!("login/logout commands never refresh")
        }

        fn to_auth<'a>(
            &'a self,
            credential: OAuthCredential,
        ) -> BoxFuture<'a, Result<ModelAuth, AuthError>> {
            Box::pin(async move {
                Ok(ModelAuth {
                    api_key: Some(credential.access),
                    ..ModelAuth::default()
                })
            })
        }
    }

    /// Recording interaction: prompts answer "entered", events accumulate.
    struct RecordingInteraction {
        events: Mutex<Vec<AuthEvent>>,
    }

    impl AuthInteraction for RecordingInteraction {
        fn signal(&self) -> Option<CancellationToken> {
            None
        }

        fn prompt(&self, _prompt: AuthPrompt) -> BoxFuture<'_, Result<String, AuthError>> {
            Box::pin(async { Ok("entered".to_string()) })
        }

        fn notify(&self, event: AuthEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn canned_credential() -> OAuthCredential {
        OAuthCredential {
            refresh: "r".to_string(),
            access: "a".to_string(),
            expires: 10,
            extra: Default::default(),
        }
    }

    #[tokio::test]
    async fn login_runs_the_flow_and_stores_the_credential_under_the_provider_id() {
        let store = crate::ai::auth::credential_store::InMemoryCredentialStore::default();
        let interaction = Arc::new(RecordingInteraction {
            events: Mutex::new(Vec::new()),
        });
        let oauth = FakeOAuth {
            credential: canned_credential(),
            received_signals: Mutex::new(Vec::new()),
        };
        let credential = login(
            &oauth,
            "anthropic",
            &store,
            interaction.clone(),
            &no_options(),
        )
        .await
        .unwrap();
        assert_eq!(credential, canned_credential());
        assert_eq!(
            store.read("anthropic", &no_options()).await.unwrap(),
            Some(Credential::OAuth(canned_credential()))
        );
        // The saved message rides the interaction's notify channel, not
        // stdio (M2d controller ruling).
        assert_eq!(
            interaction.events.lock().unwrap().clone(),
            vec![AuthEvent::Info {
                message: "Credentials saved to auth.json".to_string(),
                links: None,
            }]
        );
    }

    #[tokio::test]
    async fn login_gives_the_flow_a_fresh_signal() {
        // Upstream: `new AbortController().signal` — present, not aborted.
        let store = InMemoryStore::default();
        let interaction = Arc::new(RecordingInteraction {
            events: Mutex::new(Vec::new()),
        });
        let oauth = FakeOAuth {
            credential: canned_credential(),
            received_signals: Mutex::new(Vec::new()),
        };
        login(&oauth, "p", &store, interaction, &no_options())
            .await
            .unwrap();
        let received = oauth.received_signals.lock().unwrap().clone();
        assert_eq!(received.len(), 1);
        let signal = received.into_iter().next().unwrap().unwrap();
        assert!(!signal.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_login_does_not_store_tokens_after_the_flow_returns() {
        struct CancelAtExchange;
        impl OAuthAuth for CancelAtExchange {
            fn name(&self) -> &str {
                "CancelAtExchange"
            }
            fn login<'a>(
                &'a self,
                interaction: ProviderAuthInteraction,
            ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
                Box::pin(async move {
                    interaction.signal.cancel();
                    Ok(canned_credential())
                })
            }
            fn refresh<'a>(
                &'a self,
                _: OAuthCredential,
                _: &'a AuthOperationOptions,
            ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
                unreachable!()
            }
            fn to_auth<'a>(
                &'a self,
                _: OAuthCredential,
            ) -> BoxFuture<'a, Result<ModelAuth, AuthError>> {
                unreachable!()
            }
        }
        let store = InMemoryStore::default();
        let interaction = Arc::new(RecordingInteraction {
            events: Mutex::new(Vec::new()),
        });
        let signal = CancellationToken::new();
        let error = login_with_device_id(
            &CancelAtExchange,
            "openai",
            &store,
            interaction.clone(),
            None,
            &AuthOperationOptions::new(signal),
        )
        .await
        .unwrap_err();
        assert_eq!(error, AuthError::Cancelled);
        assert_eq!(store.read("openai", &no_options()).await.unwrap(), None);
        assert!(interaction.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_failed_login_stores_nothing_and_notifies_nothing() {
        struct FailingOAuth;
        impl OAuthAuth for FailingOAuth {
            fn name(&self) -> &str {
                "Failing"
            }
            fn login<'a>(
                &'a self,
                _interaction: ProviderAuthInteraction,
            ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
                Box::pin(async { Err(AuthError::Operation("flow aborted".to_string())) })
            }
            fn refresh<'a>(
                &'a self,
                _: OAuthCredential,
                _: &'a AuthOperationOptions,
            ) -> BoxFuture<'a, Result<OAuthCredential, AuthError>> {
                unreachable!()
            }
            fn to_auth<'a>(
                &'a self,
                _: OAuthCredential,
            ) -> BoxFuture<'a, Result<ModelAuth, AuthError>> {
                unreachable!()
            }
        }

        let store = InMemoryStore::default();
        let interaction = Arc::new(RecordingInteraction {
            events: Mutex::new(Vec::new()),
        });
        let error = login(
            &FailingOAuth,
            "p",
            &store,
            interaction.clone(),
            &no_options(),
        )
        .await
        .unwrap_err();
        assert_eq!(error, AuthError::Operation("flow aborted".to_string()));
        assert_eq!(store.read("p", &no_options()).await.unwrap(), None);
        assert!(interaction.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn logout_deletes_the_stored_credential() {
        let store = InMemoryStore::default();
        store
            .modify(
                "p",
                Box::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("k".to_string()),
                            env: None,
                            extra: Default::default(),
                        })))
                    })
                }),
                &no_options(),
            )
            .await
            .unwrap();
        logout(&store, "p", &no_options()).await.unwrap();
        assert_eq!(store.read("p", &no_options()).await.unwrap(), None);
        // Deleting an unknown provider is fine.
        logout(&store, "absent", &no_options()).await.unwrap();
    }
}
