import { useState } from "react";
import { Eye, EyeOff, LockKeyhole, Mail, UserRound, Activity } from "lucide-react";
import "./AuthScreen.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export default function AuthScreen({ onAuthenticated, onForgotPassword }) {
  const [mode, setMode] = useState("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const isRegister = mode === "register";

  function switchMode(nextMode) {
    setMode(nextMode);
    setError("");
    setSuccess("");
    setPassword("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedEmail || !password) {
      setError("Please enter your email and password.");
      return;
    }

    if (isRegister && !fullName.trim()) {
      setError("Please enter your full name.");
      return;
    }

    if (isRegister && password.length < 8) {
      setError("Your password must contain at least 8 characters.");
      return;
    }

    setLoading(true);

    try {
      const endpoint = isRegister ? "/api/auth/register" : "/api/auth/login";

      const payload = isRegister
        ? {
            email: normalizedEmail,
            password,
            full_name: fullName.trim(),
          }
        : {
            email: normalizedEmail,
            password,
          };

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
          ? detail.map((item) => item.msg).join(" ")
          : typeof detail === "string"
            ? detail
            : "The request could not be completed. Please try again.";

        throw new Error(message);
      }

      if (isRegister) {
        setSuccess("Account created successfully. You can now log in.");
        setMode("login");
        setPassword("");
      } else {
        if (!data.access_token) {
          throw new Error("The server did not return an access token.");
        }

        onAuthenticated?.(data.access_token);
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
          <span className="auth-eyebrow">AI-POWERED RETINAL ANALYSIS</span>
          <h1>
            A clearer view
            <br />
            of retinal health.
          </h1>
          <p>
            Intelligent retinal image analysis, explainable AI, and evidence-based
            clinical knowledge in one workspace.
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
        <div className="auth-form-container">
          <div className="auth-mobile-brand">
            <div className="auth-brand-icon">
              <Activity size={23} />
            </div>
            <span>RetinaAI</span>
          </div>

          <div className="auth-heading">
            <span className="auth-eyebrow">
              {isRegister ? "GET STARTED" : "WELCOME BACK"}
            </span>
            <h2>{isRegister ? "Create your account" : "Sign in to RetinaAI"}</h2>
            <p>
              {isRegister
                ? "Create an account to access your workspace."
                : "Enter your credentials to continue to your workspace."}
            </p>
          </div>

          <div className="auth-tabs" role="tablist" aria-label="Authentication">
            <button
              type="button"
              role="tab"
              aria-selected={!isRegister}
              className={!isRegister ? "active" : ""}
              onClick={() => switchMode("login")}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={isRegister}
              className={isRegister ? "active" : ""}
              onClick={() => switchMode("register")}
            >
              Create account
            </button>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {isRegister && (
              <div className="auth-field">
                <label htmlFor="auth-full-name">Full name</label>
                <div className="auth-input-wrap">
                  <UserRound size={18} aria-hidden="true" />
                  <input
                    id="auth-full-name"
                    type="text"
                    autoComplete="name"
                    placeholder="Enter your full name"
                    value={fullName}
                    onChange={(event) => setFullName(event.target.value)}
                    maxLength={150}
                    required
                  />
                </div>
              </div>
            )}

            <div className="auth-field">
              <label htmlFor="auth-email">Email address</label>
              <div className="auth-input-wrap">
                <Mail size={18} aria-hidden="true" />
                <input
                  id="auth-email"
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

            <div className="auth-field">
              <label htmlFor="auth-password">Password</label>
              <div className="auth-input-wrap">
                <LockKeyhole size={18} aria-hidden="true" />
                <input
                  id="auth-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  placeholder={
                    isRegister ? "At least 8 characters" : "Enter your password"
                  }
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={isRegister ? 8 : undefined}
                  maxLength={128}
                  required
                />
                <button
                  className="auth-password-toggle"
                  type="button"
                  onClick={() => setShowPassword((visible) => !visible)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {!isRegister && (
                <div className="auth-forgot-password">
                    <button
                        type="button"
                        onClick={onForgotPassword}
                    >
                        Forgot password?
                    </button>
                </div>
            )}

            {error && (
              <div className="auth-message auth-error" role="alert">
                {error}
              </div>
            )}

            {success && (
              <div className="auth-message auth-success" role="status">
                {success}
              </div>
            )}

            <button className="auth-submit" type="submit" disabled={loading}>
              {loading
                ? "Please wait..."
                : isRegister
                  ? "Create account"
                  : "Sign in"}
            </button>
          </form>

          <p className="auth-switch-prompt">
            {isRegister ? "Already have an account?" : "New to RetinaAI?"}{" "}
            <button
              type="button"
              onClick={() => switchMode(isRegister ? "login" : "register")}
            >
              {isRegister ? "Sign in" : "Create an account"}
            </button>
          </p>

          <p className="auth-security-note">
            Your workspace requires an authenticated account.
          </p>
        </div>
      </section>
    </main>
  );
}