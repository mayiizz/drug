import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  style?: React.CSSProperties;
}

export default function Card({ children, className = "", title, subtitle, style }: CardProps) {
  return (
    <div
      className={`rounded-2xl p-6 transition-all ${className}`}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        ...style,
      }}
    >
      {(title || subtitle) && (
        <div className="mb-5">
          {title && (
            <h3 className="font-semibold text-base" style={{ color: "var(--text-dark)" }}>
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-sm mt-0.5" style={{ color: "var(--text-light)" }}>
              {subtitle}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
