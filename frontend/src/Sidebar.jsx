import { useEffect } from "react";
import {
  Activity,
  Eye,
  MessageCircle,
  ShieldCheck,
  RefreshCw,
  ChevronRight,
  X,
} from "lucide-react";

import "./Sidebar.css";

function Sidebar({
  currentPage,
  onNavigate,
  onLogout,
  apiStatus,
  onRefreshApi,
  isOpen,
  onClose,
}) {
  // Close the sidebar when Escape is pressed.
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  // Prevent the page behind the drawer from scrolling on mobile.
  useEffect(() => {
    if (!isOpen) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  function handleNavigation(page) {
    onNavigate(page);
    onClose();
  }

  return (
    <>
      {/* Floating button shown when the sidebar is closed */}
      {!isOpen && (
        <button
          type="button"
          className="sidebar-open-button"
          onClick={() => onNavigate(null)}
          aria-label="Open sidebar"
          title="Open sidebar"
        >
          <ChevronRight size={22} />
        </button>
      )}

      {/* Dark backdrop */}
      <div
        className={`sidebar-backdrop ${isOpen ? "visible" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sidebar drawer */}
      <aside
        className={`sidebar responsive-sidebar ${
          isOpen ? "sidebar-open" : "sidebar-closed"
        }`}
        aria-label="Main navigation"
        aria-hidden={!isOpen}
        inert={!isOpen}
      >
        <div className="sidebar-top-row">
          <a
            className="brand"
            href="#/analysis"
            onClick={() => handleNavigation("analysis")}
            aria-label="RetinaAI home"
          >
            <span className="brand-mark">
              <Eye size={25} strokeWidth={2.1} />
            </span>

            <span>
              <strong>RetinaAI</strong>
              <small>DIAGNOSTIC RESEARCH</small>
            </span>
          </a>

          {/* Cross button shown when the sidebar is open */}
          <button
            type="button"
            className="sidebar-close-button"
            onClick={onClose}
            aria-label="Close sidebar"
            title="Close sidebar"
          >
            <X size={21} />
          </button>
        </div>

        <div className="sidebar-label">Workspace</div>

        <a
          className={`nav-item ${
            currentPage === "analysis" ? "active" : ""
          }`}
          href="#/analysis"
          onClick={() => handleNavigation("analysis")}
        >
          <Activity size={17} />
          <span>DR analysis</span>
        </a>

        <a
          className={`nav-item ${
            currentPage === "rag" ? "active" : ""
          }`}
          href="#/rag"
          onClick={() => handleNavigation("rag")}
        >
          <MessageCircle size={17} />
          <span>Research assistant</span>
        </a>

        <div className="sidebar-spacer" />

        <div className="sidebar-logout-section">
            <button
                type="button"
                className="nav-item auth-logout-button"
                onClick={() => {
                    onClose();
                    onLogout();
                }}
            >
                <ShieldCheck size={17} />
                <span>Log Out</span>
            </button>
        </div>

        <div className="sidebar-footer">
          <span className={`status-dot ${apiStatus}`} />

          <span>
            API {apiStatus === "checking" ? "checking" : apiStatus}
          </span>

          <button
            type="button"
            className="icon-button refresh-button"
            title="Check API status again"
            aria-label="Refresh API status"
            onClick={onRefreshApi}
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;