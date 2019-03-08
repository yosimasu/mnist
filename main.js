/*
 * variables
 */
var model;
var canvas;
var classNames = [];
var canvas;
var coords = [];
var mousePressed = false;
var mode;

/**
 * prepare the drawing canvas 
 */
$(function() {
  canvas = window._canvas = new fabric.Canvas("canvas");
  canvas.backgroundColor = "#ffffff";
  canvas.isDrawingMode = 0;
  canvas.freeDrawingBrush.color = "black";
  canvas.freeDrawingBrush.width = 10;
  canvas.renderAll();
  //setup listeners
  canvas.on("mouse:up", function(e) {
    getFrame();
    mousePressed = false;
  });
  canvas.on("mouse:down", function(e) {
    mousePressed = true;
  });
  canvas.on("mouse:move", function(e) {
    recordCoor(e);
  });
});

/**
 * allow drawing on canvas
 */
function allowDrawing() {
  canvas.isDrawingMode = 1;
  document.getElementById('status').innerHTML = 'Model Loaded';
  $('button').prop('disabled', false);
  var slider = document.getElementById('myRange');
  slider.oninput = function() {
      canvas.freeDrawingBrush.width = this.value;
  };
}

/**
 * clear the canvs 
 */
function erase() {
  canvas.clear();
  canvas.backgroundColor = '#ffffff';
  coords = [];

  $('.number').text('N/A');
  for (var i = 0; i < 10; i++) {
    $(`.p${i}`).text('N/A');
  }
}

/**
 * get the prediction 
 */
function getFrame() {
  //make sure we have at least two recorded coordinates 
  if (coords.length >= 2) {
    //get the image data from the canvas 
    const imgData = getImageData()

    //get the prediction 
    const pred = model.predict(preprocess(imgData)).dataSync()

    pred.forEach((v, i) => {
      $(`.p${i}`).text(Math.round(v * 100)/100);
    })

    $('.number').text(findIndicesOfMax(pred, 1));
  }
}

/**
 * get the current image data 
 */
function getImageData() {
  //get the minimum bounding box around the drawing 
  const mbb = getMinBox()

  //get image data according to dpi 
  const dpi = window.devicePixelRatio
  const imgData = canvas.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
                                                (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
  return imgData
}

/**
 * get indices of the top probs
 */
function findIndicesOfMax(inp, count) {
  var outp = [];
  for (var i = 0; i < inp.length; i++) {
      outp.push(i); // add index to output array
      if (outp.length > count) {
          outp.sort(function(a, b) {
              return inp[b] - inp[a];
          }); // descending sort the output array
          outp.pop(); // remove the last index (index of smallest element in output array)
      }
  }
  return outp;
}

/**
 * get the best bounding box by trimming around the drawing
 */
function getMinBox() {
  //get coordinates 
  var coorX = coords.map(function(p) {
      return p.x
  });
  var coorY = coords.map(function(p) {
      return p.y
  });

  //find top left and bottom right corners 
  var min_coords = {
      x: Math.min.apply(null, coorX),
      y: Math.min.apply(null, coorY)
  }
  var max_coords = {
      x: Math.max.apply(null, coorX),
      y: Math.max.apply(null, coorY)
  }

  //return as strucut 
  return {
      min: min_coords,
      max: max_coords
  }
}

/**
 * preprocess the data
 */
function preprocess(imgData) {
  return tf.tidy(() => {
      //convert to a tensor 
      let tensor = tf.browser.fromPixels(imgData, numChannels = 1)
      
      //resize 
      const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()

      //We add a dimension to get a batch shape 
      const batched = resized.expandDims(0)
      return batched
  })
}

/**
 * record the current drawing coordinates
 */
function recordCoor(event) {
  var pointer = canvas.getPointer(event.e);
  var posX = pointer.x;
  var posY = pointer.y;

  if (posX >= 0 && posY >= 0 && mousePressed) {
      coords.push(pointer)
  }
}

/**
 * load the model
 */
async function start() {
  //load the model 
  model = await tf.loadLayersModel('model/model.json')
  
  //warm up 
  model.predict(tf.zeros([1, 28, 28, 1]))

  //allow drawing on the canvas 
  allowDrawing()
  
  //load the class names
  // await loadDict()
}
