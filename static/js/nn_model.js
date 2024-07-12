const preprocessImage = (rawCanvas) => {
  let tensor = tf.browser
    .fromPixels(rawCanvas, numChannels=1)// Converts image to tensor with 1 color channel (grayscale)
    .resizeNearestNeighbor([28, 28])// Resizes image
    .reshape([1, 28, 28, 1]) // Reshapes tensor to match model's expected input shape
    .toFloat();
  
  return tensor.div(255.0);// Normalizes pixel values to range [0, 1]
};

function predict(rawCanvas) {
  if (window.model) {
    const image = preprocessImage(rawCanvas);
    const scores = window.model.predict(image).dataSync();
    let prediction = scores.indexOf(Math.max(...scores));
    return prediction;
  }
}

(async () => {
  window.model = await tf.loadLayersModel('model/model.json');
  console.log('Neural network model loaded!');
})();
