import { useEffect, useRef, useState } from "react";

import axios from "axios";

import {

  Activity,

  AlertCircle,

  ArrowRight,

  CheckCircle2,

  CircleHelp,

  Eye,

  FileImage,

  HeartPulse,

  LoaderCircle,

  MessageCircle,

  RefreshCw,

  ShieldCheck,

  Sparkles,

  Upload,

  X,

} from "lucide-react";

import "./App.css";

import RAGChat from "./RAGChat";

import AuthScreen from "./AuthScreen";

import Sidebar from "./Sidebar";

import ForgotPassword from "./ForgotPassword";

import ResetPassword from "./ResetPassword";

const API_BASE_URL = (

  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"

).replace(/\/$/, "");



const CLASSES = [

  { name: "No_DR", label: "No DR", color: "#159b83", description: "No diabetic retinopathy detected by the model." },

  { name: "Mild", label: "Mild", color: "#5c8ee6", description: "Mild non-proliferative changes predicted." },

  { name: "Moderate", label: "Moderate", color: "#e9a23b", description: "Moderate non-proliferative changes predicted." },

  { name: "Severe", label: "Severe", color: "#e27645", description: "Severe non-proliferative changes predicted." },

  { name: "Proliferative_DR", label: "Proliferative DR", color: "#d95370", description: "Proliferative changes predicted." },

];


function formatClass(value) {

  if (typeof value === "number") {

    return CLASSES[value]?.label ?? `Class ${value}`;

  }



  const found = CLASSES.find(

    (item) =>

      item.name.toLowerCase() === String(value ?? "").toLowerCase() ||

      item.label.toLowerCase() === String(value ?? "").toLowerCase()

  );



  return found?.label ?? String(value ?? "Unknown").replaceAll("_", " ");

}



function toPercentage(value) {

  const number = Number(value);

  if (!Number.isFinite(number)) return 0;

  return Math.max(0, Math.min(100, number <= 1 ? number * 100 : number));

}



function normalizeProbabilities(data) {

  const raw = data?.probabilities ?? data?.class_probabilities;



  if (Array.isArray(raw)) {

    return CLASSES.map((item, index) => ({

      ...item,

      value: toPercentage(raw[index] ?? 0),

    }));

  }



  if (raw && typeof raw === "object") {

    return CLASSES.map((item) => ({

      ...item,

      value: toPercentage(raw[item.name] ?? raw[item.label] ?? 0),

    }));

  }



  return CLASSES.map((item) => ({ ...item, value: 0 }));

}



function decodeImage(value) {

  if (!value) return "";

  if (value.startsWith("data:")) return value;

  return `data:image/png;base64,${value}`;

}



function App() {

  const [token,setToken] = useState(

    () => sessionStorage.getItem("retinaai_token") || ""

  );

  const[authenticated, setAuthenticated] = useState(false);

  const[authChecking, setAuthChecking] = useState(true);

  const[authError, setAuthError] = useState("");

  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);

  const [preview, setPreview] = useState("");

  const [prediction, setPrediction] = useState(null);

  const [explanation, setExplanation] = useState(null);

  const [activeImage, setActiveImage] = useState("overlay");

  const [showForgotPassword, setShowForgotPassword] = useState(false);

  const [showResetPassword, setShowResetPassword] = useState(false);
  const [resetToken, setResetToken] = useState("");

  const [dragging, setDragging] = useState(false);

  const [loading, setLoading] = useState(false);

  const [apiStatus, setApiStatus] = useState("checking");

  const [error, setError] = useState("");

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [currentPage, setCurrentPage] = useState(()=> {

    const hash= window.location.hash;



    if (hash === "#/rag"){

      return "rag";

    }



    return "analysis";

  });



  useEffect(() => {

    function handleHashChange() {

      const hash = window.location.hash;

      if (hash === "#/rag") {

        setCurrentPage("rag");

      } else {

        setCurrentPage("analysis");

      }

      //close the sidebar whenevr the route changes
      setSidebarOpen(false);

    }



    window.addEventListener("hashchange", handleHashChange);



    //set the default route when no hash is present

    if (!window.location.hash) {

      window.location.hash = "#/analysis";

    }



    return () => {

      window.removeEventListener("hashchange", handleHashChange);

    };

  }, []);



  useEffect(() => {

    let cancelled = false;



    async function checkApi() {

      try {

        // FastAPI normally exposes its OpenAPI schema at /openapi.json.

        await axios.get(`${API_BASE_URL}/openapi.json`, { timeout: 5000 });

        if (!cancelled) setApiStatus("online");

      } catch {

        if (!cancelled) setApiStatus("offline");

      }

    }



    checkApi();

    return () => {

      cancelled = true;

    };

  }, []);



  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    let cancelled = false;

    async function verifySession() {
      if (!token) {
        if (!cancelled) {
          setAuthenticated(false);
          setAuthChecking(false);
        }
        return;
      }

      try {
        await axios.get(`${API_BASE_URL}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!cancelled) {
          setAuthenticated(true);
        }
      } catch {
        sessionStorage.removeItem("retinaai_token");
        if (!cancelled) {
          setToken("");
          setAuthenticated(false);
        }
      } finally {
        if (!cancelled) {
          setAuthChecking(false);
        }
      }
    }

    setAuthChecking(true);
    verifySession();

    return () => {
      cancelled = true;
    };
  }, [token]);

  function handleAuthenticated(accessToken) {

    sessionStorage.setItem("retinaai_token", accessToken);

    setToken(accessToken);

    setAuthenticated(true);


  }



  function handleLogout() {

    sessionStorage.removeItem("retinaai_token");

    setToken("");

    setAuthenticated(false);

    setCurrentPage("analysis");

    window.location.hash = "#/analysis";

  }

  function cloaseSidebar() {
    setSidebarOpen(false);
  }

  function navigateToPage(page) {
    //Passing null opens the sidebar from the floating arrow.
    if(page=== null){
      setSidebarOpen(true);
      return;
    }

    //Change page and close the sidebar
    setCurrentPage(page);
    setSidebarOpen(false);

    const targetHash = page === "rag" ? "#/rag" : "#/analysis";

    if (window.location.hash !== targetHash) {
      window.location.hash = targetHash;
    }
  }

  async function refreshApiStatus() {
    setApiStatus("checking");

    try {
      await axios.get(`${API_BASE_URL}/openapi.json`, {
        timeout: 5000,
      });

      setApiStatus("online");
    } catch {
      setApiStatus("offline");
    }
  }

  function acceptFile(selectedFile) {

    setError("");

    setPrediction(null);

    setExplanation(null);

    setActiveImage("overlay");



    if (!selectedFile) return;



    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];



    if (!allowedTypes.includes(selectedFile.type)) {

      setError("Please select a JPG, PNG, or WebP image.");

      return;

    }



    if (selectedFile.size > 10 * 1024 * 1024) {

      setError("The image is too large. Please select a file smaller than 10 MB.");

      return;

    }



    setFile(selectedFile);

  }



  function handleDrop(event) {

    event.preventDefault();

    setDragging(false);

    acceptFile(event.dataTransfer.files?.[0]);

  }



  async function analyzeImage() {

    if (!file) {

      setError("Choose a retinal image before starting the analysis.");

      return;

    }



    setLoading(true);

    setError("");

    setPrediction(null);

    setExplanation(null);



    const formData = new FormData();

    formData.append("image", file);



    try {

      // Step 1: request the classification.

      const predictionResponse = await axios.post(

        `${API_BASE_URL}/api/predictions/`,

        formData,

        {
          timeout: 120000,
          headers: { Authorization: `Bearer ${token}` },
        }

      );



      const predictionData = predictionResponse.data;

      setPrediction(predictionData);



      // Step 2: request Grad-CAM for the uploaded image.

      // The explanation endpoint also returns the model's prediction.

      try {

        const explanationForm = new FormData();

        explanationForm.append("image", file);



        const explanationResponse = await axios.post(

          `${API_BASE_URL}/api/explanations/`,

          explanationForm,

          {
            timeout: 120000,
            headers: { Authorization: `Bearer ${token}` },
          }

        );



        setExplanation(explanationResponse.data);

      } catch (explanationError) {

        console.error("Grad-CAM request failed:", explanationError);

        setError(

          "The classification completed, but the Grad-CAM explanation could not be generated. You can still view the prediction."

        );

      }

    } catch (requestError) {

      console.error("Prediction request failed:", requestError);



      if (requestError.response) {

        const detail = requestError.response.data?.detail;

        setError(

          typeof detail === "string"

            ? detail

            : `The API returned HTTP ${requestError.response.status}. Check the backend logs and API request schema.`

        );

      } else if (requestError.code === "ECONNABORTED") {

        setError("The request took too long. Please try again.");

      } else {

        setError(

          "Could not reach the backend. Make sure FastAPI is running at " +

            API_BASE_URL +

            "."

        );

      }

    } finally {

      setLoading(false);

    }

  }



  function resetAnalysis() {

    setFile(null);

    setPrediction(null);

    setExplanation(null);

    setError("");

    setActiveImage("overlay");



    if (fileInputRef.current) fileInputRef.current.value = "";

  }



  const predictedValue =

    prediction?.predicted_class ??

    prediction?.class_name ??

    prediction?.predicted_class_index ??

    prediction?.class_index;



  const confidence = toPercentage(

    prediction?.confidence ?? prediction?.class_score ?? 0

  );



  const imageContext = prediction

    ? {

        predicted_class: prediction.predicted_class,

        predicted_class_index: prediction.predicted_class_index,

        confidence: prediction.confidence,

        probabilities: prediction.probabilities,

        model_name: prediction.model_name,

        explanation: explanation

          ? {

              explained_class: explanation.explained_class ?? null,

              explained_class_index:

                explanation.explained_class_index ?? null,

              target_layer: explanation.target_layer ?? null,

              available: true,

          }

          : null,

    }

    : null;



  const probabilities = normalizeProbabilities(prediction);

  const highestProbability = Math.max(...probabilities.map((item) => item.value));



  const explanationImage =

    activeImage === "overlay"

      ? decodeImage(

          explanation?.overlay_png_base64 ?? explanation?.overlay_base64

        )

      : decodeImage(

          explanation?.heatmap_png_base64 ?? explanation?.heatmap_base64

        );



  const hasResults = Boolean(prediction);



  if (authChecking) {
    return (
      <div className="auth-loading-screen" role="status">
        Checking your RetinaAI session...
      </div>
    );
  }

  if (!authenticated) {
    if (showResetPassword) {
      return (
        <ResetPassword
          token={resetToken}
          onBackToLogin={() => {
            setShowResetPassword(false);
            setResetToken("");
            setShowForgotPassword(false);
          }}
          onResetSuccess={() => {
            setShowResetPassword(false);
            setResetToken("");
            setShowForgotPassword(false);
          }}
        />
      );
    }

    if (showForgotPassword) {
      return (
        <ForgotPassword
          onBackToLogin={() => setShowForgotPassword(false)}
          onResetLink={(token) => {
            setResetToken(token);
            setShowResetPassword(true);
          }}
        />
      );
    }

    return (
      <AuthScreen
        onAuthenticated={handleAuthenticated}
        onForgotPassword={() => setShowForgotPassword(true)}
      />
    );
  }

  return (

    <div className="app-shell">

      <Sidebar
        currentPage={currentPage}
        onNavigate={navigateToPage}
        onLogout={handleLogout}
        apiStatus={apiStatus}
        onRefreshApi={refreshApiStatus}
        isOpen={sidebarOpen}
        onClose={cloaseSidebar}
      />

      <main className="main-content" id="home">

        {currentPage === "analysis" && (

          <>

            <header className="topbar">

              <div className="breadcrumb">

                Workspace <span>/</span> <strong>Image analysis</strong>

              </div>

            </header>



            <section className="page-heading" id="analysis">

              <div>

                <div className="eyebrow"><HeartPulse size={15} /> AI-POWERED RETINAL RESEARCH</div>

                <h1>Diabetic retinopathy<br className="mobile-break" /> analysis</h1>

                <p className="page-description">

                  Upload a retinal fundus image to explore the model's classification

                  and a visual explanation of its output.

                </p>

              </div>

              <div className="heading-decoration" aria-hidden="true">

                <div className="orbit orbit-one" />

                <div className="orbit orbit-two" />

                <Eye size={43} strokeWidth={1.2} />

              </div>

            </section>



            <div className="workflow-steps">

              <div className="workflow-step completed">

                <span>01</span>

                <div><strong>Upload image</strong><small>Choose a retinal scan</small></div>

              </div>

              <ArrowRight className="step-arrow" size={17} />

              <div className={`workflow-step ${hasResults ? "completed" : ""}`}>

                <span>02</span>

                <div><strong>Analyze</strong><small>AI classification</small></div>

              </div>

              <ArrowRight className="step-arrow" size={17} />

              <div className={`workflow-step ${explanation ? "completed" : ""}`}>

                <span>03</span>

                <div><strong>Explore</strong><small>Visual explanation</small></div>

              </div>

            </div>



            <div className="workspace-grid">

              <section className="panel upload-panel">

                <div className="panel-heading">

                  <div>

                    <span className="section-kicker">STEP 01</span>

                    <h2>Retinal image</h2>

                  </div>

                  {file && (

                    <button className="text-button" onClick={resetAnalysis}>

                      <X size={14} /> Clear

                    </button>

                  )}

                </div>



                <input

                  ref={fileInputRef}

                  className="visually-hidden"

                  type="file"

                  accept="image/jpeg,image/png,image/webp"

                  onChange={(event) => acceptFile(event.target.files?.[0])}

                />



                <div

                  className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}

                  onDragOver={(event) => {

                    event.preventDefault();

                    setDragging(true);

                  }}

                  onDragLeave={() => setDragging(false)}

                  onDrop={handleDrop}

                  onClick={() => !file && fileInputRef.current?.click()}

                  role={file ? undefined : "button"}

                  tabIndex={file ? undefined : 0}

                  onKeyDown={(event) => {

                    if (!file && (event.key === "Enter" || event.key === " ")) {

                      event.preventDefault();

                      fileInputRef.current?.click();

                    }

                  }}

                >

                  {preview ? (

                    <>

                      <img className="uploaded-preview" src={preview} alt="Selected retinal image preview" />

                      <div className="file-overlay">

                        <CheckCircle2 size={17} />

                        <span>Image ready for analysis</span>

                      </div>

                      <button

                        className="remove-image"

                        type="button"

                        aria-label="Remove image"

                        onClick={(event) => {

                          event.stopPropagation();

                          resetAnalysis();

                        }}

                      >

                        <X size={17} />

                      </button>

                    </>

                  ) : (

                    <div className="drop-content">

                      <div className="upload-icon"><Upload size={24} /></div>

                      <h3>Drop your retinal image here</h3>

                      <p>or click to browse files</p>

                      <span className="file-types">JPG, PNG or WEBP · MAX 10 MB</span>

                    </div>

                  )}

                </div>



                {file && (

                  <div className="file-details">

                    <div className="file-icon"><FileImage size={19} /></div>

                    <div className="file-name">

                      <strong>{file.name}</strong>

                      <span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Image selected</span>

                    </div>

                    <CheckCircle2 size={18} className="file-check" />

                  </div>

                )}



                <div className="upload-tip">

                  <CircleHelp size={16} />

                  <p>Use a clear, properly exposed retinal fundus photograph for the best input quality.</p>

                </div>



                <button

                  className="primary-button analyze-button"

                  onClick={analyzeImage}

                  disabled={!file || loading}

                >

                  {loading ? (

                    <><LoaderCircle className="spin" size={18} /> Analyzing image...</>

                  ) : (

                    <><Sparkles size={17} /> Analyze retinal image <ArrowRight size={17} /></>

                  )}

                </button>



                {error && (

                  <div className="error-message" role="alert">

                    <AlertCircle size={18} />

                    <span>{error}</span>

                  </div>

                )}



                <div className="privacy-note">

                  <ShieldCheck size={16} />

                  <span>Images are sent to your configured backend for analysis.</span>

                </div>

              </section>



              <section className="panel results-panel">

                <div className="panel-heading">

                  <div>

                    <span className="section-kicker">STEP 02</span>

                    <h2>Classification results</h2>

                  </div>

                  {hasResults && (

                    <div className="results-action">

                      <span className="results-tag">

                        <CheckCircle2 size={14} /> Complete

                      </span>

                      <button

                        type="button"

                        className="new-analysis-button"

                        onClick={resetAnalysis}

                        disabled={loading}

                      >

                        <RefreshCw size={14} /> New analysis

                      </button>

                    </div>

                  )}

                </div>



                {!hasResults ? (

                  <div className="empty-results">

                    <div className="empty-illustration">

                      <div className="empty-ring ring-a" />

                      <div className="empty-ring ring-b" />

                      <Eye size={37} strokeWidth={1.3} />

                    </div>

                    <h3>Your results will appear here</h3>

                    <p>

                      Upload a retinal image and run the analysis to see the predicted

                      class and the model's scores.

                    </p>

                    <div className="empty-hint"><span /> Waiting for image analysis</div>

                  </div>

                ) : (

                  <div className="results-content">

                    <div className="prediction-summary">

                      <div className="prediction-label">MODEL PREDICTION</div>

                      <div className="prediction-title-row">

                        <h3>{formatClass(predictedValue)}</h3>

                        <span

                          className="prediction-indicator"

                          style={{

                            background:

                              CLASSES.find((item) => item.name === predictedValue)?.color ??

                              CLASSES.find((item) => item.label === formatClass(predictedValue))?.color ??

                              "#159b83",

                          }}

                        />

                      </div>

                      <p>

                        {CLASSES.find(

                          (item) =>

                            item.name === predictedValue ||

                            item.label === formatClass(predictedValue)

                        )?.description ?? "The model returned a classification for this image."}

                      </p>

                      <div className="confidence-row">

                        <span>Model confidence score</span>

                        <strong>{confidence.toFixed(2)}%</strong>

                      </div>

                      <div className="confidence-track">

                        <div style={{ width: `${confidence}%` }} />

                      </div>

                      <small className="confidence-caption">

                        A model score is not a measure of clinical certainty.

                      </small>

                    </div>



                    <div className="probability-section">

                      <div className="subsection-heading">

                        <h3>Class score distribution</h3>

                        <span>5 classes</span>

                      </div>

                      <div className="probability-list">

                        {probabilities.map((item) => (

                          <div className="probability-item" key={item.name}>

                            <div className="probability-label">

                              <span>{item.label}</span>

                              <strong>{item.value.toFixed(2)}%</strong>

                            </div>

                            <div className="probability-track">

                              <div

                                className="probability-fill"

                                style={{

                                  width: `${item.value}%`,

                                  background: item.color,

                                }}

                              />

                            </div>

                          </div>

                        ))}

                      </div>

                      {highestProbability === 0 && (

                        <p className="data-note">

                          Probability scores were not found in the API response. Check

                          the response field names in the backend.

                        </p>

                      )}

                    </div>



                    <div className="model-meta">

                      <div>

                        <span>Classes</span>

                        <strong>5 DR categories</strong>

                      </div>

                    </div>

                  </div>

                )}

              </section>

            </div>



            <section className="panel explain-panel">

              <div className="panel-heading explain-heading">

                <div>

                  <span className="section-kicker">STEP 03</span>

                  <h2>Visual explanation</h2>

                  <p className="panel-subtitle">

                    Grad-CAM highlights image regions that influenced the model's output.

                  </p>

                </div>

                {explanation && (

                  <span className="method-badge"><Sparkles size={14} /> Grad-CAM</span>

                )}

              </div>



              {!explanation ? (

                <div className="explain-empty">

                  <div className="explain-empty-icon"><Sparkles size={23} /></div>

                  <div>

                    <h3>Explore what influenced the prediction</h3>

                    <p>

                      Run an image analysis to request a heatmap and an overlay from

                      the explanation endpoint.

                    </p>

                  </div>

                </div>

              ) : (

                <div className="explanation-content">

                  <div className="explanation-tabs" role="tablist" aria-label="Explanation image">

                    <button

                      className={activeImage === "overlay" ? "selected" : ""}

                      onClick={() => setActiveImage("overlay")}

                      role="tab"

                      aria-selected={activeImage === "overlay"}

                    >

                      Overlay

                    </button>

                    <button

                      className={activeImage === "heatmap" ? "selected" : ""}

                      onClick={() => setActiveImage("heatmap")}

                      role="tab"

                      aria-selected={activeImage === "heatmap"}

                    >

                      Heatmap

                    </button>

                  </div>



                  <div className="gradcam-image-wrap">

                    {explanationImage ? (

                      <img

                        src={explanationImage}

                        alt={activeImage === "overlay" ? "Grad-CAM overlay visualization" : "Grad-CAM heatmap visualization"}

                      />

                    ) : (

                      <div className="image-unavailable">

                        <AlertCircle size={20} />

                        <span>The selected explanation image was not present in the API response.</span>

                      </div>

                    )}

                  </div>



                  <div className="gradcam-caption">

                    <Sparkles size={16} />

                    <span>

                      Explained class: <strong>{formatClass(

                        explanation?.explained_class ??

                        explanation?.explained_class_index ??

                        predictedValue

                      )}</strong>

                    </span>

                    <span className="gradcam-layer">

                      {explanation?.target_layer ?? "features[-1]"}

                    </span>

                  </div>

                </div>

              )}



              <div className="explanation-disclaimer">

                <AlertCircle size={16} />

                <p>

                  Grad-CAM is a model-interpretation aid, not proof that highlighted

                  areas contain disease or that the prediction is correct.

                </p>

              </div>

            </section>



            <section className="info-grid" id="how-it-works">

              <div className="info-card">

                <div className="info-icon teal"><Eye size={19} /></div>

                <h3>Image-based analysis</h3>

                <p>A retinal image is processed by the trained classification model.</p>

              </div>

              <div className="info-card">

                <div className="info-icon blue"><Activity size={19} /></div>

                <h3>Five DR categories</h3>

                <p>The model outputs scores for five diabetic retinopathy classes.</p>

              </div>

              <div className="info-card">

                <div className="info-icon purple"><Sparkles size={19} /></div>

                <h3>Visual explanation</h3>

                <p>Grad-CAM provides a visualization of influential model activations.</p>

              </div>

            </section>



            <footer className="medical-disclaimer">

              <div className="disclaimer-icon"><ShieldCheck size={20} /></div>

              <div>

                <strong>Important medical disclaimer</strong>

                <p>

                  This platform is an experimental research prototype, not a medical

                  device or diagnostic service. Results may be inaccurate and must not

                  be used alone to make clinical decisions. Consult a qualified

                  eye-care professional for diagnosis and treatment.

                </p>

              </div>

            </footer>



            <div className="page-footer">

              <span>RetinaAI <span className="footer-separator">·</span> DR research platform</span>

              <span>For research and educational use</span>

            </div>

          </>

        )}



        {currentPage === "rag" && (

          <RAGChat imageContext={imageContext} token={token} />

        )}

      </main>

    </div>

  );

}



export default App;