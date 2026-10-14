import { useCallback, useState } from "react";

type AspectPreviewProps = {
  src: string;
  alt: string;
  className?: string;
  /** Cap the longer edge in CSS pixels while keeping aspect ratio. */
  maxEdge?: number;
  onClick?: (e: React.MouseEvent) => void;
  asButton?: boolean;
  caption?: string;
  draggable?: boolean;
};

/**
 * Image preview sized to the image's natural aspect ratio.
 * Uses onLoad naturalWidth/Height — no server metadata required.
 */
export function AspectPreview({
  src,
  alt,
  className = "",
  maxEdge,
  onClick,
  asButton,
  caption,
  draggable = false,
}: AspectPreviewProps) {
  const [ratio, setRatio] = useState<number | null>(null);

  const onLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      setRatio(img.naturalWidth / img.naturalHeight);
    }
  }, []);

  // Reset ratio when src changes so we don't keep a stale AR.
  const [prevSrc, setPrevSrc] = useState(src);
  if (src !== prevSrc) {
    setPrevSrc(src);
    setRatio(null);
  }

  const style: React.CSSProperties = {};
  if (ratio != null) {
    style.aspectRatio = String(ratio);
    if (maxEdge != null && maxEdge > 0) {
      if (ratio >= 1) {
        style.maxWidth = "100%";
        style.width = "100%";
        // Cap height via max-width when landscape; for portrait use maxHeight
      } else {
        style.maxHeight = maxEdge;
        style.width = "100%";
      }
      if (ratio >= 1) {
        // Landscape: width drives height; cap via maxHeight derived from AR
        style.maxHeight = maxEdge / ratio;
      }
    }
  }

  const img = (
    <img
      src={src}
      alt={alt}
      draggable={draggable}
      onLoad={onLoad}
      className="dl-aspect-preview-img"
      style={ratio == null ? { height: "auto" } : undefined}
    />
  );

  const body = (
    <div className={`dl-aspect-preview ${className}`.trim()} style={style}>
      {img}
    </div>
  );

  if (asButton || onClick) {
    return (
      <button
        type="button"
        className="dl-aspect-preview-btn nodrag nopan"
        onClick={onClick}
      >
        {body}
        {caption ? <span className="dl-aspect-preview-cap">{caption}</span> : null}
      </button>
    );
  }

  return body;
}
