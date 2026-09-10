// Make sure OrbitControls is loaded from CDN or included if you use it
// If OrbitControls is loaded globally via script tag: const OrbitControls = window.OrbitControls;
// const TWEEN = window.TWEEN; // If using TWEEN CDN

// --- Global Variables ---
let scene, camera, renderer, controls;
const objectsInScene = new Map(); // Stores THREE.Mesh objects by their ID
const websocketUrl = "ws://localhost:8765";
let socket;

// --- DOM Elements ---
const statusElement = document.getElementById('status');
const taskDescriptionElement = document.getElementById('taskDescription');
const visualizationContainer = document.getElementById('visualizationContainer');

// --- Initialization ---
function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xdddddd);

    // Camera
    camera = new THREE.PerspectiveCamera(75, visualizationContainer.clientWidth / visualizationContainer.clientHeight, 0.1, 1000);
    camera.position.set(3, 4, 5); // Adjusted camera position
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(visualizationContainer.clientWidth, visualizationContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true; // Enable shadows
    visualizationContainer.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7);
    directionalLight.castShadow = true; // Enable shadow casting for this light
    // Configure shadow properties for better quality (optional)
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    scene.add(directionalLight);

    // Ground Plane
    const planeGeometry = new THREE.PlaneGeometry(20, 20);
    const planeMaterial = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.8 });
    const groundPlane = new THREE.Mesh(planeGeometry, planeMaterial);
    groundPlane.rotation.x = -Math.PI / 2;
    groundPlane.receiveShadow = true; // Allow plane to receive shadows
    scene.add(groundPlane);

    // Controls (Optional, if OrbitControls is loaded)
    if (typeof OrbitControls !== 'undefined') {
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 2;
        controls.maxDistance = 20;
        controls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent camera from going below ground
    }

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Start animation loop
    animate();

    // Connect to WebSocket
    connectWebSocket();
}

// --- WebSocket Handling ---
function connectWebSocket() {
    socket = new WebSocket(websocketUrl);
    statusElement.textContent = "Connecting to WebSocket...";

    socket.onopen = () => {
        statusElement.textContent = "Connected to Physics Server!";
        console.log("WebSocket connected.");
        // You could send a "client_ready" message if needed
    };

    socket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            // console.log("Message from server:", message);
            if (message.type === "scene_update" || message.type === "initial_scene") {
                updateScene(message.payload); // payload is List[ObjectData]
                if (message.type === "initial_scene" && message.task_description) {
                    taskDescriptionElement.textContent = message.task_description;
                }
            } else if (message.type === "task_info") { // Example for updating task description
                taskDescriptionElement.textContent = message.description || "N/A";
            }
        } catch (e) {
            console.error("Error processing message from server:", e, event.data);
        }
    };

    socket.onerror = (error) => {
        statusElement.textContent = "WebSocket Error!";
        console.error("WebSocket Error:", error);
    };

    socket.onclose = () => {
        statusElement.textContent = "Disconnected. Attempting to reconnect in 3s...";
        console.log("WebSocket disconnected. Reconnecting in 3 seconds...");
        setTimeout(connectWebSocket, 3000); // Simple reconnect logic
    };
}

// --- Three.js Scene Updates ---
function updateScene(objectStates) { // objectStates is List of Dicts from server
    const receivedIds = new Set();

    objectStates.forEach(objState => {
        receivedIds.add(objState.id);
        let threeObject = objectsInScene.get(objState.id);

        if (!threeObject) { // Object doesn't exist, create it
            let geometry;
            const scale = objState.scale || [1,1,1];
            if (objState.type === "cube") {
                geometry = new THREE.BoxGeometry(scale[0], scale[1], scale[2]);
            } else if (objState.type === "sphere") {
                geometry = new THREE.SphereGeometry(scale[0] / 2, 32, 16); // Assume scale[0] is diameter
            } else {
                console.warn("Unsupported object type for visualization:", objState.type);
                geometry = new THREE.BoxGeometry(1, 1, 1); // Default placeholder
            }

            const color = new THREE.Color(...(objState.color_rgba ? objState.color_rgba.slice(0,3) : [0.5, 0.5, 0.5]));
            const material = new THREE.MeshStandardMaterial({ color: color, roughness: 0.5, metalness: 0.1 });
            threeObject = new THREE.Mesh(geometry, material);
            threeObject.name = objState.id; // Useful for debugging
            threeObject.castShadow = true; // Object casts shadows
            threeObject.receiveShadow = false; // Usually objects don't receive shadows on themselves unless complex

            scene.add(threeObject);
            objectsInScene.set(objState.id, threeObject);
        }

        // Update position and orientation
        if (objState.position) {
            threeObject.position.set(...objState.position);
        }
        if (objState.orientation_quaternion) {
            threeObject.quaternion.set(...objState.orientation_quaternion);
        }
        // TODO: Update color or other properties if they can change dynamically
    });

    // Remove objects that are in Three.js scene but not in the new state
    objectsInScene.forEach((obj, id) => {
        if (!receivedIds.has(id)) {
            scene.remove(obj);
            obj.geometry.dispose(); // Dispose geometry
            obj.material.dispose(); // Dispose material
            objectsInScene.delete(id);
            console.log(`Removed object ${id} from scene.`);
        }
    });
}

// --- Animation Loop & Resize ---
function animate() {
    requestAnimationFrame(animate);
    if (controls) {
        controls.update(); // Only if OrbitControls is used
    }
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = visualizationContainer.clientWidth / visualizationContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(visualizationContainer.clientWidth / visualizationContainer.clientHeight);
}

// --- Start Everything ---
init();
