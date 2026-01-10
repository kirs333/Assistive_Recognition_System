function StatusBar({ mode, fps, objectCount, activeObject }) {
  return (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-label">Mode:</span>
        <span className={`status-value mode-${mode.toLowerCase()}`}>
          {mode}
        </span>
      </div>

      <div className="status-item">
        <span className="status-label">FPS:</span>
        <span className="status-value">{fps.toFixed(1)}</span>
      </div>

      <div className="status-item">
        <span className="status-label">Objects:</span>
        <span className="status-value">{objectCount}</span>
      </div>

      {activeObject && (
        <div className="status-item active-object-status">
          <span className="status-label">Active:</span>
          <span className="status-value active-object-name">
            {activeObject}
          </span>
        </div>
      )}
    </div>
  );
}

export default StatusBar;
