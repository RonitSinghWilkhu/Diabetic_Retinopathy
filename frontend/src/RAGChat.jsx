import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; 
import axios from "axios";
import {
  Bot, BookOpen, Check, ChevronRight, CircleHelp, Copy, FileText,
  LoaderCircle, MessageCircle, RotateCcw, Send, ShieldCheck, Sparkles,
  Trash2, UserRound, ExternalLink,
} from "lucide-react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
function getPDFUrl(filename, pages = []) {
  const baseUrl = `${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}`;
  const page = pages.find((value) => Number.isInteger(Number(value)) && Number(value) > 0);

  return page ? `${baseUrl}?page=${page}` : baseUrl;
}
const SUGGESTIONS = [
  "What are the clinical stages of diabetic retinopathy?",
  "What are the major risk factors for diabetic retinopathy?",
  "How is diabetic retinopathy screened and graded?",
  "What evidence supports AI-based DR screening?",
];

function normaliseSources(sources = []) {
  const seen = new Set();
  return sources.map((source, index) => {
    const item = typeof source === "string"
      ? (() => {
          const match = source.match(/^(.*?)\s*\(pages:\s*(.*?)\)$/i);
          return {
            filename: match ? match[1] : source,
            pages: match && match[2] ? match[2].split(/,\s*/).map((page) => Number(page)).filter(Number.isFinite) : [],
            evidence_id: index + 1,
            category: "Research source",
          };
        })()
      : {
          evidence_id: source.evidence_id ?? index + 1,
          filename: source.filename ?? source.source ?? source.document ?? "Unknown source",
          pages: source.pages ?? [],
          category: source.category ?? "Document",
          excerpt: source.excerpt ?? source.text ?? "",
          document_url: source.document_url ?? source.url ?? "",
        };
    const key = `${item.filename}|${(item.pages || []).join(",")}`;
    if (seen.has(key)) return null;
    seen.add(key);
    return { ...item, pages: Array.isArray(item.pages) ? item.pages : [] };
  }).filter(Boolean);
}

function AnswerText({ text, onCitation }) {
  // Convert evidence markers to links that ReactMarkdown can render as buttons.
  const markdown = String(text || "").replace(
    /\\[Evidence\\s+(\\d+)\\]/gi,
    (_, id) => `[Evidence ${id}](evidence:${id})`
  );

  return (
    <div className="rag-answer-text markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href = "", children, ...props }) => {
            const match = href.match(/^evidence:(\\d+)$/i);
            if (match) {
              return (
                <button
                  type="button"
                  className="evidence-citation"
                  onClick={() => onCitation(Number(match[1]))}
                  aria-label={`View evidence source ${match[1]}`}
                >
                  {children}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                {children}
              </a>
            );
          },
          table: ({ children, ...props }) => (
            <div className="markdown-table-wrap">
              <table {...props}>{children}</table>
            </div>
          ),
          code: ({ children, className, ...props }) => {
            const isBlock = Boolean(className);
            return isBlock
              ? <code className={className} {...props}>{children}</code>
              : <code className="markdown-inline-code" {...props}>{children}</code>;
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}

export default function RAGChat({ imageContext = null }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [activeEvidence, setActiveEvidence] = useState(null);
  const [mobileTab, setMobileTab] = useState("chat");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const selectedSources = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    return lastAssistant?.sources || [];
  }, [messages]);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, pending, error]);

  async function sendQuestion(value = question, retryMessage = null) {
    const clean = String(value || "").trim();
    if (!clean || pending) return;
    setError("");
    setQuestion("");
    setPending(true);
    if (!retryMessage) setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: clean }]);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat/`, { question: clean, image_context: imageContext }, { timeout: 120000 });
      const data = response.data || {};
      const sources = normaliseSources(data.sources || []);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: "assistant",
        text: data.answer || "I couldn't generate an answer from the available evidence.",
        sources, disclaimer: data.disclaimer || "This information is for educational purposes only.",
        question: clean,
      }]);
      setActiveEvidence(sources[0]?.evidence_id ?? null);
      setMobileTab("chat");
      requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }));
    } catch (requestError) {
      const detail = requestError.response?.data?.detail;
      setError(typeof detail === "string" ? detail : `Couldn't reach the chat API at ${API_BASE_URL}. Check that the FastAPI server is running.`);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function clearChat() {
    setMessages([]); setError(""); setActiveEvidence(null); setQuestion("");
  }

  async function copyAnswer(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError("Copying was blocked by the browser. Select the answer text and copy it manually.");
    }
  }

  const hasConversation = messages.length > 0;
  const activeMessage = [...messages].reverse().find((message) => message.role === "assistant");
  const evidence = selectedSources.find((source) => Number(source.evidence_id) === Number(activeEvidence));

  return <section className="rag-workspace">
    <header className="rag-topbar">
      <div className="rag-title-lockup"><span className="rag-title-icon"><Sparkles size={19} /></span><div><h1>DR Research Assistant</h1><p>Evidence-grounded diabetic retinopathy education</p></div></div>
      <div className="rag-topbar-actions"><span className="rag-grounded-badge"><span /> Evidence mode</span><button className="rag-icon-button" onClick={clearChat} disabled={!hasConversation || pending} title="Clear conversation" aria-label="Clear conversation"><Trash2 size={17} /></button></div>
    </header>
    <div className="rag-mobile-tabs"><button className={mobileTab === "chat" ? "active" : ""} onClick={() => setMobileTab("chat")}> <MessageCircle size={16}/> Chat</button><button className={mobileTab === "evidence" ? "active" : ""} onClick={() => setMobileTab("evidence")}><BookOpen size={16}/> Evidence {selectedSources.length > 0 && <span>{selectedSources.length}</span>}</button></div>
    <div className="rag-panels">
      <div className={`rag-chat-panel ${mobileTab === "chat" ? "mobile-active" : "mobile-hidden"}`}>
        <div className="rag-message-list" ref={scrollRef}>
          {!hasConversation && <div className="rag-welcome"><div className="rag-welcome-symbol"><Bot size={28}/></div><span className="rag-eyebrow">YOUR EVIDENCE-BASED ASSISTANT</span><h2>Explore the science behind<br/><em>diabetic retinopathy.</em></h2><p>Ask questions about clinical stages, screening guidelines, risk factors, or research evidence. Answers are grounded in your indexed documents.</p><div className="rag-suggestion-grid">{SUGGESTIONS.map((item) => <button key={item} onClick={() => sendQuestion(item)} disabled={pending}><span>{item}</span><ChevronRight size={16}/></button>)}</div></div>}
          {messages.map((message, index) => <article className={`rag-message ${message.role}`} key={message.id}>
            <div className={`rag-avatar ${message.role}`}>{message.role === "assistant" ? <Bot size={17}/> : <UserRound size={17}/>}</div>
            <div className="rag-message-content"><div className="rag-message-meta"><strong>{message.role === "assistant" ? "DR Research Assistant" : "You"}</strong>{message.role === "assistant" && <span>Evidence-grounded response</span>}</div>
              {message.role === "assistant" ? <><AnswerText text={message.text} onCitation={(id) => { setActiveEvidence(id); setMobileTab("evidence"); } }/><div className="rag-answer-footer"><button onClick={() => copyAnswer(message.text)}><Copy size={14}/> Copy answer</button>{message.sources.length > 0 && <button onClick={() => setMobileTab("evidence")}><BookOpen size={14}/> {message.sources.length} sources</button>}<button onClick={() => sendQuestion(message.question, message)} disabled={pending}><RotateCcw size={14}/> Retry</button></div><p className="rag-answer-disclaimer"><ShieldCheck size={13}/>{message.disclaimer}</p></> : <p className="rag-user-text">{message.text}</p>}
            </div>
          </article>)}
          {pending && <div className="rag-message assistant"><div className="rag-avatar assistant"><Bot size={17}/></div><div className="rag-message-content"><div className="rag-message-meta"><strong>DR Research Assistant</strong><span>Searching evidence</span></div><div className="rag-thinking"><LoaderCircle size={16}/> Reviewing documents and preparing an answer…</div></div></div>}
          {error && <div className="rag-error"><CircleHelp size={18}/><div><strong>Couldn't complete that request</strong><p>{error}</p><button onClick={() => { setError(""); inputRef.current?.focus(); }}>Try again</button></div></div>}
        </div>
        <form className="rag-composer" onSubmit={(event) => { event.preventDefault(); sendQuestion(); }}><div className="rag-composer-box"><textarea ref={inputRef} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); } }} placeholder="Ask a question about diabetic retinopathy…" rows={2} maxLength={2000} disabled={pending} aria-label="Your question"/><button type="submit" disabled={pending || !question.trim()} aria-label="Send question">{pending ? <LoaderCircle size={18} className="rag-spin"/> : <Send size={18}/>}</button></div><div className="rag-composer-hint"><span><ShieldCheck size={13}/> Educational use only · Not a diagnostic tool</span><span>{question.length}/2000 · Enter to send</span></div></form>
      </div>
      <aside className={`rag-evidence-panel ${mobileTab === "evidence" ? "mobile-active" : "mobile-hidden"}`}>
        <div className="rag-evidence-heading"><div><span className="rag-eyebrow">SOURCE LIBRARY</span><h2><BookOpen size={19}/> Evidence panel</h2></div><span className="rag-source-count">{selectedSources.length}</span></div>
        {!activeMessage ? <div className="rag-evidence-empty"><div><FileText size={25}/></div><h3>Evidence will appear here</h3><p>Ask a question to see the documents used to generate the answer, including source names and page references.</p><div className="rag-evidence-note"><ShieldCheck size={16}/><span>Always check the original source before relying on a medical claim.</span></div></div> : selectedSources.length === 0 ? <div className="rag-evidence-empty"><div><FileText size={25}/></div><h3>No sources returned</h3><p>The API did not return source metadata for this answer.</p></div> : <><p className="rag-evidence-intro">Documents returned for the latest answer. Select a source to inspect its details.</p><div className="rag-source-list">{selectedSources.map((source, index) => <button key={`${source.filename}-${index}`} className={`rag-source-card ${Number(activeEvidence) === Number(source.evidence_id) ? "selected" : ""}`} onClick={() => setActiveEvidence(source.evidence_id)}><span className="rag-source-card-top"><span className="rag-source-file-icon"><FileText size={17}/></span><span className="rag-source-type">{source.category || "RESEARCH SOURCE"}</span><span className="rag-source-arrow"><ChevronRight size={16}/></span></span><a href={getPDFUrl(source.filename, source.pages)} target="_blank" rel="noopener noreferrer" className="rag-source-filename" onClick={(event) => event.stopPropagation()} title="Open original PDF">{source.filename}<ExternalLink size={13} /></a><span className="rag-source-meta">{source.pages?.length ? `Pages ${source.pages.join(", ")}` : "Page details not provided"}<span>Evidence {source.evidence_id}</span></span>{source.excerpt && <p>{source.excerpt}</p>}</button>)}</div>{evidence && <div className="rag-evidence-detail"><span className="rag-eyebrow">SELECTED SOURCE</span><h3>{evidence.filename}</h3>{evidence.pages?.length > 0 && <p className="rag-detail-pages">Pages: {evidence.pages.join(", ")}</p>}{evidence.excerpt ? <blockquote>{evidence.excerpt}</blockquote> : <p className="rag-detail-placeholder">The current API returns source names and pages only. A source excerpt will appear here once the backend provides structured evidence.</p>}{evidence.document_url && <a href={evidence.document_url} target="_blank" rel="noreferrer" className="rag-open-document">Open original PDF <ExternalLink size={14}/></a>}</div>}</>}
        <div className="rag-evidence-footer"><ShieldCheck size={15}/><p>Retrieved sources are references, not a substitute for professional clinical judgement.</p></div>
      </aside>
    </div>
  </section>;
}
