import { useEffect, useRef } from "react";

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

function VideoDisplay({ frameData }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!frameData || !frameData.image) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;

    img.src = `data:image/jpeg;base64,${frameData.image}`;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (frameData.detections && frameData.detections.length > 0) {
        frameData.detections.forEach((detection) => {
          const [xmin, ymin, xmax, ymax] = detection.bbox;
          const color = BBOX_COLORS[detection.class_id % BBOX_COLORS.length];

          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

          const label = `${detection.class}: ${detection.confidence.toFixed(
            2
          )}`;
          ctx.font = "16px Arial";
          const textWidth = ctx.measureText(label).width;
          const textHeight = 20;

          ctx.fillStyle = color;
          ctx.fillRect(
            xmin,
            ymin - textHeight - 5,
            textWidth + 10,
            textHeight + 5
          );

          ctx.fillStyle = "white";
          ctx.fillText(label, xmin + 5, ymin - 8);
        });
      }

      if (frameData.active_bbox && frameData.mode === "GUIDE") {
        const [xmin, ymin, xmax, ymax] = frameData.active_bbox;

        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 5;
        ctx.strokeRect(xmin - 2, ymin - 2, xmax - xmin + 4, ymax - ymin + 4);

        ctx.fillStyle = "#00FF00";
        ctx.fillRect(xmin, ymax + 5, 80, 25);
        ctx.fillStyle = "black";
        ctx.font = "bold 16px Arial";
        ctx.fillText("ACTIVE", xmin + 5, ymax + 22);
      }
    };
  }, [frameData]);

  return (
    <div className="video-container">
      <div className="video-wrapper">
        <img
          ref={imgRef}
          className="video-frame"
          alt="Video feed"
          style={{ display: "none" }}
        />
        <canvas ref={canvasRef} className="video-canvas" />
      </div>
      {!frameData && (
        <div className="no-video-overlay">
          <p>Waiting for video stream...</p>
        </div>
      )}
    </div>
  );
}

export default VideoDisplay;
