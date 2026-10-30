import { useState } from "react";
import { Activity, ArrowLeft, LockKeyhole } from "lucide-react";
import "./ResetPassword.css";
import "./AuthScreen.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export default function ResetPassword({ token, onBackToLogin, onResetSuccess }) {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!token) {
      setError("Reset token is missing. Generate a new reset link.");
      return;
    }

    if (password !== confirmPassword) {
      setError("The passwords do not match.");
      return;
    }

    if (password.length < 8 || password.length > 128) {
      setError("Password must be between 8 and 128 characters.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          new_password: password,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to reset your password. Generate a new reset link."
        );
      }

      setSuccess("Password reset successfully. You can now sign in.");
      setPassword("");
      setConfirmPassword("");

      if (onResetSuccess) {
        onResetSuccess();
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
        <div className="reset-form-container">
          <div className="reset-mobile-brand">
            <div className="auth-brand-icon">
              <Activity size={23} />
            </div>
            <span>RetinaAI</span>
          </div>

          <div className="auth-heading">
            <span className="auth-eyebrow">ACCOUNT SECURITY</span>
            <h2>Reset your password</h2>
            <p>Choose a new password for your RetinaAI account.</p>
          </div>

          <form className="reset-form" onSubmit={handleSubmit}>
            <div className="reset-field">
              <label htmlFor="reset-password">New password</label>

              <div className="reset-input-wrap">
                <LockKeyhole size={18} aria-hidden="true" />
                <input
                  id="reset-password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="At least 8 characters"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={8}
                  maxLength={128}
                  required
                />
              </div>
            </div>

            <div className="reset-field">
              <label htmlFor="reset-confirm-password">
                Confirm new password
              </label>

              <div className="reset-input-wrap">
                <LockKeyhole size={18} aria-hidden="true" />
                <input
                  id="reset-confirm-password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Enter your new password again"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  minLength={8}
                  maxLength={128}
                  required
                />
              </div>
            </div>

            {error && (
              <div className="reset-message reset-error" role="alert">
                {error}
              </div>
            )}

            {success && (
              <div className="reset-message reset-success" role="status">
                {success}
              </div>
            )}

            <button
              className="reset-submit"
              type="submit"
              disabled={loading}
            >
              {loading ? "Resetting password..." : "Reset password"}
            </button>
          </form>

          <button
            className="reset-back"
            type="button"
            onClick={onBackToLogin}
          >
            <ArrowLeft size={16} />
            Back to sign in
          </button>
        </div>
      </section>
    </main>
  );
}