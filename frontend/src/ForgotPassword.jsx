import { useState } from "react";
import { Activity, ArrowLeft, Mail } from "lucide-react";
import "./ForgotPassword.css";
import "./AuthScreen.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export default function ForgotPassword({ onBackToLogin, onResetLink }) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [resetToken, setResetToken] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");
    setResetToken("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/auth/forgot-password`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: email.trim().toLowerCase(),
          }),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to process the request. Please try again."
        );
      }

      setSuccess(
        "If this email belongs to a registered account, a reset link has been generated for development."
      );

      if (data.reset_token) {
        setResetToken(data.reset_token);
      }
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Unable to connect to RetinaAI. Check that the backend is running."
          : err.message || "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleOpenResetLink() {
    if (resetToken && onResetLink) {
      onResetLink(resetToken);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-brand-panel">
        <div className="auth-brand">
          <div className="auth-brand-icon">
            <Activity size={25} strokeWidth={2.2} />
          </div>
          <span>RetinaAI</span>
        </div>

        <div className="auth-brand-content">
          <span className="auth-eyebrow">
            AI-POWERED RETINAL ANALYSIS
          </span>

          <h1>
            A clearer view
            <br />
            of retinal health.
          </h1>

          <p>
            Intelligent retinal image analysis, explainable AI, and
            evidence-based clinical knowledge in one workspace.
          </p>

          <div className="auth-feature-list">
            <div>
              <span className="auth-feature-dot" />
              Diabetic retinopathy classification
            </div>
            <div>
              <span className="auth-feature-dot" />
              Visual AI explanations
            </div>
            <div>
              <span className="auth-feature-dot" />
              Research-informed assistance
            </div>
          </div>
        </div>

        <p className="auth-disclaimer">
          Research prototype · Not a substitute for professional medical advice
        </p>
      </section>

      <section className="auth-form-panel">
        <div className="forgot-form-container">
          <div className="forgot-mobile-brand">
            <div className="auth-brand-icon">
              <Activity size={23} />
            </div>
            <span>RetinaAI</span>
          </div>

          <div className="auth-heading">
            <span className="auth-eyebrow">ACCOUNT RECOVERY</span>
            <h2>Forgot your password?</h2>
            <p>
              Enter the email address associated with your account.
              We'll help you reset your password.
            </p>
          </div>

          <form className="forgot-form" onSubmit={handleSubmit}>
            <div className="forgot-field">
              <label htmlFor="forgot-email">Email address</label>

              <div className="forgot-input-wrap">
                <Mail size={18} aria-hidden="true" />
                <input
                  id="forgot-email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  maxLength={255}
                  required
                />
              </div>
            </div>

            {error && (
              <div className="forgot-message forgot-error" role="alert">
                {error}
              </div>
            )}

            {success && (
              <div className="forgot-message forgot-success" role="status">
                {success}
              </div>
            )}

            {resetToken && (
              <div className="forgot-dev-note">
                <p>
                  Development reset link is ready. Click below to continue.
                </p>
                <button
                  className="forgot-submit"
                  type="button"
                  onClick={handleOpenResetLink}
                >
                  Open reset password screen
                </button>
              </div>
            )}

            <button
              className="forgot-submit"
              type="submit"
              disabled={loading}
            >
              {loading ? "Please wait..." : "Generate reset link"}
            </button>
          </form>

          <button
            className="forgot-back"
            type="button"
            onClick={onBackToLogin}
          >
            <ArrowLeft size={16} />
            Back to sign in
          </button>

          <p className="forgot-dev-note forgot-footer-note">
            Development mode: reset links are generated locally, not emailed.
          </p>
        </div>
      </section>
    </main>
  );
}
