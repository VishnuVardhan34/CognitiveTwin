import React, { useEffect, useRef } from 'react';

const ALERT_STYLES = {
  overload: { border: '#f78166', bg: 'rgba(247,129,102,0.12)', icon: '⚠️' },
  fatigue:  { border: '#d29922', bg: 'rgba(210,153,34,0.12)',  icon: '😴' },
  default:  { border: '#58a6ff', bg: 'rgba(88,166,255,0.12)',  icon: 'ℹ️' },
};

export default function AlertOverlay({ alert, onDismiss }) {
  const style = ALERT_STYLES[alert?.type] || ALERT_STYLES.default;
  const overlayRef = useRef(null);

  useEffect(() => {
    // Auto-dismiss after 8 seconds
    const timer = setTimeout(() => onDismiss?.(), 8000);
    return () => clearTimeout(timer);
  }, [alert, onDismiss]);

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') onDismiss?.();
  };

  return (
    <div
      className="alert-backdrop"
      onClick={onDismiss}
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-modal="true"
      aria-label="Cognitive state alert"
      tabIndex={-1}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000,
        animation: 'fadeIn 0.25s ease',
      }}
    >
      <div
        ref={overlayRef}
        onClick={e => e.stopPropagation()}
        style={{
          background: '#21262d',
          border: `2px solid ${style.border}`,
          backgroundColor: style.bg,
          borderRadius: '12px',
          padding: '32px',
          maxWidth: '420px',
          width: '90%',
          textAlign: 'center',
          boxShadow: `0 0 32px ${style.border}44`,
          animation: 'fadeIn 0.25s ease',
        }}
      >
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>{style.icon}</div>
        <h2 style={{ color: style.border, marginBottom: '12px', fontSize: '18px', fontWeight: 700 }}>
          {alert?.type === 'overload' ? 'Cognitive Overload' : 'Fatigue Detected'}
        </h2>
        <p style={{ color: '#e6edf3', marginBottom: '24px', fontSize: '14px', lineHeight: '1.6' }}>
          {alert?.message}
        </p>
        <button
          onClick={onDismiss}
          style={{
            background: style.border,
            color: '#0d1117',
            border: 'none',
            borderRadius: '6px',
            padding: '10px 24px',
            fontSize: '14px',
            fontWeight: 700,
            cursor: 'pointer',
          }}
        >
          Acknowledge
        </button>
      </div>
    </div>
  );
}
