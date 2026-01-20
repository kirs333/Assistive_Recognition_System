import { useEffect, useRef } from "react";


/// Color for bounding boxes
const BBOX_COLORS = [
  "#A47857",
  "#4494E4",
  "#5D61D1",
  "#B2B685",
  "#589F6A",
  "#60CAE7",
  "#9F7CA8",
  "#A9A2F1",
  "#627696",
  "#ACB0B8",
];


/// draw detected object boxes in canvas
function drawDetections(ctx, detections) {
  detections.forEach((det) => {
    const [xmin, ymin, xmax, ymax] = det.bbox;
    const color = BBOX_COLORS[det.class_id % BBOX_COLORS.length];


    /// draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);


    /// draw label
    const label = `${det.class} ${det.confidence.toFixed(2)}`;
    ctx.font = "14px Inter, Arial, sans-serif";

    const padding = 6;
    const textWidth = ctx.measureText(label).width;
    const textHeight = 18;

    ctx.fillStyle = color;
    ctx.fillRect(
      xmin,
      ymin - textHeight - padding,
      textWidth + padding * 2,
      textHeight + padding
    );

    ctx.fillStyle = "#fff";
    ctx.fillText(label, xmin + padding, ymin - padding);
  });
}

function drawActiveBox(ctx, bbox) {
  const [xmin, ymin, xmax, ymax] = bbox;

  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 4;
  ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

  ctx.fillStyle = "#00FF00";
  ctx.fillRect(xmin, ymax + 6, 72, 22);

  ctx.fillStyle = "#000";
  ctx.font = "bold 13px Inter, Arial, sans-serif";
  ctx.fillText("ACTIVE", xmin + 8, ymax + 22);
}

function VideoDisplay({ frameData }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!frameData?.image) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;

    img.src = `data:image/jpeg;base64,${frameData.image}`;

    img.onload = () => {
      const dpr = window.devicePixelRatio || 1;

      canvas.width = img.width * dpr;
      canvas.height = img.height * dpr;
      canvas.style.width = `${img.width}px`;
      canvas.style.height = `${img.height}px`;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      if (frameData.detections?.length) {
        drawDetections(ctx, frameData.detections);
      }

      if (frameData.mode === "GUIDE" && frameData.active_bbox) {
        drawActiveBox(ctx, frameData.active_bbox);
      }
    };
  }, [frameData]);

  return (
    <div className="video-container">
      <div className="video-wrapper">
        <img ref={imgRef} alt="" style={{ display: "none" }} />
        <canvas ref={canvasRef} className="video-canvas" />
      </div>

      {!frameData && (
        <div className="no-video-overlay">
          <span>Waiting for video stream…</span>
        </div>
      )}
    </div>
  );
}

export default VideoDisplay;
