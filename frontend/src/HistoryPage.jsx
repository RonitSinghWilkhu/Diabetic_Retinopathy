import { useEffect, useState } from "react";
import { AlertCircle, CalendarDays, FileImage, RefreshCw, ShieldCheck } from "lucide-react";
import "./HistoryPage.css";

const LABELS = { No_DR: "No DR", Mild: "Mild", Moderate: "Moderate", Severe: "Severe", Proliferative_DR: "Proliferative DR" };
const dateLabel = (v) => { const d = new Date(v); return v && !Number.isNaN(d.getTime()) ? d.toLocaleString() : "Date unavailable"; };
const percent = (v) => Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(2)}%` : "—";

function ImagePanel({ title, src, missing }) {
  return <div className="history-image-panel"><div className="history-image-title">{title}</div>
    {src ? <img src={src} alt={title} loading="lazy" /> : <div className="history-image-missing"><FileImage size={23}/><span>{missing}</span></div>}
  </div>;
}

export default function HistoryPage({ token, apiBaseUrl }) {
  const API_BASE_URL = (
    apiBaseUrl || import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
  ).replace(/\/$/, "");
  const [items, setItems] = useState([]);
  const [images, setImages] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const objectUrls = [];
    async function load() {
      setLoading(true); setError(""); setImages({});
      if (!token) { setError("Sign in to view saved analyses."); setLoading(false); return; }
      try {
        const response = await fetch(`${API_BASE_URL}/api/history/`, { headers: { Authorization: `Bearer ${token}` } });
        if (!response.ok) throw new Error(response.status === 401 ? "Your session has expired. Sign in again." : "Could not load analysis history.");
        const rows = await response.json();
        if (cancelled) return;
        setItems(Array.isArray(rows) ? rows : []);
        const pairs = await Promise.all((Array.isArray(rows) ? rows : []).map(async row => {
          const record = { original: null, heatmap: null, overlay: null };
          await Promise.all([
            ["original", `${API_BASE_URL}/api/history/${row.id}/image`],
            ["heatmap", `${API_BASE_URL}/api/history/${row.id}/artifacts/heatmap`],
            ["overlay", `${API_BASE_URL}/api/history/${row.id}/artifacts/overlay`]
          ].map(async ([key, url]) => {
            try {
              const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
              if (!r.ok) return;
              const objectUrl = URL.createObjectURL(await r.blob());
              objectUrls.push(objectUrl); record[key] = objectUrl;
            } catch { /* Missing files display placeholders. */ }
          }));
          return [row.id, record];
        }));
        if (!cancelled) setImages(Object.fromEntries(pairs));
      } catch (e) { if (!cancelled) setError(e.message || "Could not load analysis history."); }
      finally { if (!cancelled) setLoading(false); }
    }
    load();
    return () => { cancelled = true; objectUrls.forEach(URL.revokeObjectURL); };
  }, [token, API_BASE_URL, refresh]);

  return <section className="history-page">
    <div className="history-heading"><div><div className="history-eyebrow"><ShieldCheck size={15}/> PRIVATE RECORDS</div>
      <h1>View Past Analysis</h1><p>Review saved retinal images, predictions, and Grad-CAM visualizations.</p></div>
      <button className="history-refresh" disabled={loading} onClick={() => setRefresh(v => v + 1)}><RefreshCw size={16}/> Refresh</button>
    </div>
    {error && <div className="history-error"><AlertCircle size={18}/>{error}</div>}
    {loading ? <div className="history-state"><RefreshCw size={22}/> Loading saved analyses…</div> :
      !error && items.length === 0 ? <div className="history-empty"><FileImage size={32}/><h2>No saved analyses yet</h2><p>Completed image analyses will appear here.</p></div> :
      <div className="history-list">{items.map(item => {
        const img = images[item.id] || {};
        const probs = item.probabilities && typeof item.probabilities === "object" ? Object.entries(item.probabilities).sort((a,b) => Number(b[1])-Number(a[1])) : [];
        return <article className="history-card" key={item.id}>
          <div className="history-card-header"><div><span className="history-record-id">ANALYSIS #{item.id}</span>
            <h2>{LABELS[item.predicted_class] || item.predicted_class || "Unknown prediction"}</h2>
            <div className="history-meta"><CalendarDays size={14}/><span>{dateLabel(item.created_at)}</span><span>•</span><span className="history-filename" title={item.original_filename}>{item.original_filename}</span></div>
          </div><div className="history-confidence"><span>Confidence</span><strong>{percent(item.confidence)}</strong></div></div>
          <div className="history-images-grid">
            <ImagePanel title="Original upload" src={img.original} missing="Original image unavailable"/>
            <ImagePanel title="Grad-CAM heatmap" src={img.heatmap} missing="Heatmap not available"/>
            <ImagePanel title="Overlay" src={img.overlay} missing="Overlay not available"/>
          </div>
          {probs.length > 0 && <details className="history-probabilities"><summary>View class probabilities</summary><div className="history-probability-list">
            {probs.map(([name,value]) => <div className="history-probability" key={name}><span>{LABELS[name] || name.replaceAll("_"," ")}</span><div className="history-probability-track"><div style={{width:`${Math.max(0,Math.min(100,Number(value)*100||0))}%`}}/></div><strong>{percent(value)}</strong></div>)}
          </div></details>}
          <div className="history-disclaimer">Grad-CAM shows model sensitivity, not confirmed disease localization or a clinical diagnosis. Model: {item.model_name || "Unknown"}.</div>
        </article>;
      })}</div>}
  </section>;
}
