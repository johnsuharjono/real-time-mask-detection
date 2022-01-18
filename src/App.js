// Import dependencies
import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";

function App() {
	const webcamRef = useRef(null);
	const canvasRef = useRef(null);

	const names = ["incorrectly", "mask", "nomask"];
	const font = "16px sans-serif";

	const videoConstraints = {
		width: 750,
		height: 500,
		facingMode: "user",
	};

	const runModel = async () => {
		const net = await tf.loadGraphModel("./web_model/model.json");
		console.log(net);

		//  Loop and detect face
		setInterval(() => {
			detect(net);
		}, 10);
	};

	const detect = async (net) => {
		// Check data is available
		if (
			typeof webcamRef.current !== "undefined" &&
			webcamRef.current !== null &&
			webcamRef.current.video.readyState === 4
		) {
			// Get Video Properties
			const video = webcamRef.current.video;
			const videoWidth = webcamRef.current.video.videoWidth;
			const videoHeight = webcamRef.current.video.videoHeight;

			// Set video width
			webcamRef.current.video.width = videoWidth;
			webcamRef.current.video.height = videoHeight;

			// Set canvas height and width
			canvasRef.current.width = videoWidth;
			canvasRef.current.height = videoHeight;

			// Make detections
			const img = tf.browser.fromPixels(video); // take img from the camera
			const processed = tf.image
				.resizeBilinear(img, [320, 320])
				.div(255.0) // normalize
				.expandDims(0);
			const obj = await net.executeAsync(processed);

			const [boxes, scores, classes, valid_detections] = obj;
			const boxes_data = boxes.dataSync();
			const scores_data = scores.dataSync();
			const classes_data = classes.dataSync();
			console.log(classes_data[0]);
			const valid_detections_data = valid_detections.dataSync()[0];

			tf.dispose(obj);

			// Draw code below
			const c = canvasRef.current;
			const ctx = c.getContext("2d");

			var i;
			for (i = 0; i < valid_detections_data; ++i) {
				let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
				x1 *= c.width;
				x2 *= c.width;
				y1 *= c.height;
				y2 *= c.height;
				const width = x2 - x1;
				const height = y2 - y1;
				const klass = names[classes_data[i]];
				const score = scores_data[i].toFixed(2);

				// Draw the bounding box.
				ctx.strokeStyle = "#00FFFF";
				ctx.lineWidth = 4;
				ctx.strokeRect(x1, y1, width, height);

				// Draw the label background.
				ctx.fillStyle = "#00FFFF";
				const textWidth = ctx.measureText(klass + ":" + score).width;
				const textHeight = parseInt(font, 10); // base 10
				ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);
			}
			for (i = 0; i < valid_detections_data; ++i) {
				let [x1, y1, ,] = boxes_data.slice(i * 4, (i + 1) * 4);
				x1 *= c.width;
				y1 *= c.height;
				const klass = names[classes_data[i]];
				const score = scores_data[i].toFixed(2);

				// Draw the text last to ensure it's on top.
				ctx.fillStyle = "#000000";
				ctx.fillText(klass + ":" + score, x1, y1 + 13);
			}

			tf.dispose(boxes);
			tf.dispose(scores);
			tf.dispose(classes);
			tf.dispose(valid_detections);
			tf.dispose(img);
			tf.dispose(processed);
			tf.dispose(boxes_data);
			tf.dispose(scores_data);
			tf.dispose(classes_data);
		}
	};

	useEffect(() => {
		runModel();
	}, []);

	return (
		<div className="App">
			<h1>Real Time Mask Detection with yolov5 and Tensorflow JS</h1>
			<div className="App-header">
				<Webcam
					className="webcam"
					ref={webcamRef}
					videoConstraints={videoConstraints}
					muted={true}
					style={{
						position: "absolute",
						marginLeft: "auto",
						marginRight: "auto",
						left: 0,
						right: 0,
						textAlign: "center",
						zindex: 9,
					}}
				/>

				<canvas
					className="canvas"
					ref={canvasRef}
					style={{
						position: "absolute",
						marginLeft: "auto",
						marginRight: "auto",
						left: 0,
						right: 0,
						textAlign: "center",
						zindex: 8,
					}}
				/>
			</div>
		</div>
	);
}

export default App;
