const ADMIN_USERNAME = "admin";
const ADMIN_PASSWORD = "password123";

function login() {
    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;
    let errorMessage = document.getElementById("error-message");

    if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
        document.getElementById("login-container").classList.add("hidden");
        document.getElementById("dashboard-container").classList.remove("hidden");
    } else {
        errorMessage.textContent = "Invalid username or password!";
    }
}

function logout() {
    document.getElementById("dashboard-container").classList.add("hidden");
    document.getElementById("prediction-container").classList.add("hidden");
    document.getElementById("login-container").classList.remove("hidden");
}

function openExcel(filePath) {
    window.location.href = filePath;
}

function showPrediction() {
    document.getElementById("dashboard-container").classList.add("hidden");
    document.getElementById("prediction-container").classList.remove("hidden");
}

// Function to navigate to SGPA Input Page
function goToSGPAInput() {
    window.location.href = "sgpa_input.html";
}

// Function to go back from SGPA Input Page to Admin Dashboard
function goBack() {
    window.location.href = "index.html";  // Now it correctly navigates back to the Admin Dashboard
}



function predictPerformance() {
    let sgpaInput = document.getElementById("sgpa-input").value.trim();

    if (!sgpaInput) {
        alert("Please enter at least one SGPA value!");
        return;
    }

    let sgpaValues = sgpaInput.split(",").map(value => parseFloat(value.trim())).filter(value => !isNaN(value));

    if (sgpaValues.length < 1) {  // Allow 1 SGPA input instead of 2
        alert("Please enter at least one SGPA value.");
        return;
    }

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sgpa_values: sgpaValues })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            localStorage.setItem("predicted_cgpa", data.predicted_cgpa);
            localStorage.setItem("graph_url", data.graph_url);
            window.location.href = "results.html";
        }
    })
    .catch(error => {
        alert("Prediction failed.");
        console.error("Fetch error:", error);
    });
}

