function addToChat(sender, message) {
  var chatBox = document.getElementById("chat-box");
  var newMessage = document.createElement("div");
  newMessage.classList.add("speech-bubble");

  var iconHTML = '';

  if (sender === "user") {
    newMessage.classList.add("user-message");
    iconHTML = '<img src="/static/images/user.jpg" alt="User Icon" class="user-icon">';
  } else {
    newMessage.classList.add("bot-message");
    iconHTML = '<img src="/static/images/bot.png" alt="Bot Icon" class="bot-icon">';
  }

  // Replace \n with <br> tags for line breaks
  message = message.replace(/\\n/g, '<br>');

  newMessage.innerHTML = iconHTML + message;
  chatBox.appendChild(newMessage);
}









function handleImageUpload(event) {
    const file = event.target.files[0]; // Get the uploaded file
    const reader = new FileReader(); // Initialize a FileReader object

    // Define what to do when the file is loaded
    reader.onload = function () {
        const imageSrc = reader.result; // Get the image data URL
        displayImage(imageSrc); // Call a function to display the image
    };

    // Read the uploaded file as a data URL
    reader.readAsDataURL(file);
}

function displayImage(imageSrc) {
  const chatBox = document.getElementById("chat-box");

  // Create a container div to hold the user icon and the image
  const containerDiv = document.createElement("div");
  containerDiv.style.overflow = "auto"; // Ensure contents wrap properly
  containerDiv.style.clear = "both"; // Clear any previous floats

  // Create a div for the user icon
  const userIconDiv = document.createElement("div");
  userIconDiv.style.width = "20px"; // Set the width of the icon
  userIconDiv.style.height = "20px"; // Set the height of the icon
  userIconDiv.style.background = "url('/static/images/remove.png')"; // Set the icon image URL
  userIconDiv.style.backgroundSize = "cover"; // Adjust background size as needed
  userIconDiv.style.float = "left"; // Float the icon to the left
  userIconDiv.style.marginRight = "10px"; // Add margin to the right of the icon

  // Create an image element
  const img = document.createElement("img");
  img.src = imageSrc; // Set the image source
  img.style.width = "200px"; // Set the width (adjust value as needed)
  img.style.height = "auto"; // Maintain aspect ratio; adjust height accordingly

  // Append the user icon and the image to the container div
  containerDiv.appendChild(userIconDiv);
  containerDiv.appendChild(img);

  // Append the container div to the chat box
  chatBox.appendChild(containerDiv);
}




// Attach the handleImageUpload function to the file input element
document.getElementById("file-input").addEventListener("change", handleImageUpload);

document.getElementById("user-input").addEventListener("keydown", function(event) {
  if (event.key === "Enter") {
    event.preventDefault(); // Prevent default form submission behavior
    sendMessage(); // Call the sendMessage function to handle the form submission
  }
});
// Function to send message or upload image
function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  
  // Check if user input is an image
  var fileInput = document.getElementById('file-input');
  var file = fileInput.files[0];
  
  if (file) {
    // If user uploaded an image, send it for processing
    sendImage(file);
    return;
  }

  // If user input is text, send it for processing
  if (userInput.trim() !== "") {
    sendText(userInput);
  }
}

// Function to send text message for processing
function sendText(userInput) {
  addToChat("user", userInput);
  document.getElementById("user-input").value = "";
  
  // Send user input to server for text processing
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/process_input", true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.onload = function () {
    if (xhr.status === 200) {
      var responseData = JSON.parse(xhr.responseText);
      addToChat("bot", responseData.response);
      speakResponse(responseData.response);
    } else {
      addToChat("bot", "Error processing request.");
    }
  };

  xhr.send(JSON.stringify({
    input: userInput
  }));
}

function sendImage(imageFile) {
  var formData = new FormData();
  formData.append('file', imageFile);

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (response.ok) {
      return response.json();
    } else {
      throw new Error('No match found for the uploaded image');
    }
  })
  .then(data => {
    if (data.pest_type) {
      addToChat("bot", "Predicted Pest Name: " + data.pest_type);
    } else {
      throw new Error('No match found for the uploaded image');
    }
  })
  .catch(error => {
    console.error('Error:', error);
    addToChat("bot", error.message);
  })
  .finally(() => {
    // Clear the image upload section
    document.getElementById('file-input').value = '';
    
    // Remove the uploaded image preview
    var uploadPreview = document.getElementById('upload-preview');
    if (uploadPreview) {
      uploadPreview.innerHTML = '';
    }
  });
}





// Wrap each letter of the text in a span element
/*const animatedText = document.getElementById("animated-text");
animatedText.innerHTML = animatedText.textContent.replace(/\S/g, "<span>$&</span>");

// Apply animation to each span element
let index = 0;
const spans = document.querySelectorAll("#animated-text span");
const intervalId = setInterval(() => {
    spans[index].classList.add("show");
    index++;
    if (index === spans.length) clearInterval(intervalId); // Stop the interval when all letters are animated
}, 50); // Adjust the interval duration as needed*/

var i = 0;
var txt = "Hi! I'm here to assist you with questions about soil , pest, and market linkage.Feel free to ask about soil testing techniques, ways to improve soil quality, or any other soil-related questions. You can also upload images of pest-affected crops to know the pest name and request suitable pesticides. Additionally, I can provide current market information to assist you in selling your agricultural products.Let me know how can I help?";

var speed = 50;

(function typeWriter() {
  if (i < txt.length) {
    document.getElementById("demo").innerHTML += txt.charAt(i);
    i++;
    setTimeout(typeWriter, speed);
  }
})();

function triggerFileInput() {
  document.getElementById('file-input').click();
}
// Function to speak out the response
function speakResponse(response) {
  var synth = window.speechSynthesis;
  
  // Check if there's an ongoing speech
  if (synth.speaking) {
    // Stop the ongoing speech
    synth.cancel();
  }
  
  var utterance = new SpeechSynthesisUtterance(response);
  utterance.lang = 'en-US';
  synth.speak(utterance);
}


// Function to handle speech recognition
var recognition;

function startSpeechRecognition() {
  if (!('webkitSpeechRecognition' in window)) {
    alert("Speech recognition is not supported by this browser");
  } else {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = function(event) {
      var speechResult = event.results[0][0].transcript;
      document.getElementById("user-input").value = speechResult;
      sendMessage();
    };
    recognition.onerror = function(event) {
      console.error('Speech recognition error:', event.error);
    };
  }
}
function switchToTamil() {
  // Redirect to the Tamil bot's index page
  window.location.href = "http://localhost:5001/"; // Replace <TAMIL_PORT> with the port of Tamil bot
}
