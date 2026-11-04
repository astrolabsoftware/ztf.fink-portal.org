// Quadrillage longitude / latitude tous les 10 degres
function makeGridTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 1440;
    canvas.height = 720;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#777777";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;

    ctx.beginPath();

    for (let lon = 0; lon <= 360; lon += 10) {
        ctx.moveTo((lon / 360) * canvas.width, 0);
        ctx.lineTo((lon / 360) * canvas.width, canvas.height);
    }

    for (let lat = -90; lat <= 90; lat += 10) {
        ctx.moveTo(0, ((lat + 90) / 180) * canvas.height);
        ctx.lineTo(canvas.width, ((lat + 90) / 180) * canvas.height);
    }

    ctx.stroke();

    return new THREE.CanvasTexture(canvas);
}

// Conversion de la geometrie Three.js en texte OBJ
function geometryToOBJ(geometry) {
    const positions = geometry.getAttribute("position");
    const normals = geometry.getAttribute("normal");
    const uvs = geometry.getAttribute("uv");
    const indices = geometry.getIndex();

    const lines = ["# SOCCA ellipsoid"];

    for (let i = 0; i < positions.count; i++) {
        lines.push(`v ${positions.getX(i)} ${positions.getY(i)} ${positions.getZ(i)}`);
    }
    if (uvs) {
        for (let i = 0; i < uvs.count; i++) {
            lines.push(`vt ${uvs.getX(i)} ${uvs.getY(i)}`);
        }
    }
    if (normals) {
        for (let i = 0; i < normals.count; i++) {
            lines.push(`vn ${normals.getX(i)} ${normals.getY(i)} ${normals.getZ(i)}`);
        }
    }

    // les indices OBJ commencent a 1
    const vertexIndex = i => (indices ? indices.getX(i) : i) + 1;
    const count = indices ? indices.count : positions.count;

    for (let i = 0; i < count; i += 3) {
        const a = vertexIndex(i), b = vertexIndex(i + 1), c = vertexIndex(i + 2);
        lines.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}`);
    }

    return lines.join("\n") + "\n";
}


// Lumiere ambiante + Soleil directionnel
function addLights(scene, sunPosition) {
    scene.add(new THREE.AmbientLight(0xffffff, 0.15));

    const sunLight = new THREE.DirectionalLight(0xffffff, 2.5);
    sunLight.position.copy(sunPosition);
    scene.add(sunLight);
}


window.dash_clientside = Object.assign({}, window.dash_clientside, {
    model3d: {
        draw: function (p) {
            if (!p || !p.a_b || !p.a_c) return "";

            const container = document.getElementById("model-3d");
            if (!container) return "";
            container.innerHTML = "";

            const w = container.clientWidth, h = container.clientHeight;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);

            const camera = new THREE.PerspectiveCamera(25, w / h, 0.1, 1000);
            let zoom = 3;
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(w, h);
            container.appendChild(renderer.domElement);

            // Plein ecran + adaptation du rendu a la taille du conteneur
            const view = document.querySelector(".model3d-view");
            document.getElementById("model3d-fullscreen").onclick = function () {
                if (!document.fullscreenElement) view.requestFullscreen();
                else document.exitFullscreen();
            };
            document.addEventListener("fullscreenchange", function () {
                renderer.setSize(container.clientWidth, container.clientHeight);
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.render(scene, camera);
            });

            // Ellipsoide SOCCA
            // SphereGeometry utilise Y comme axe polaire, SOCCA utilise Z comme axe de rotation
            const geometry = new THREE.SphereGeometry(1, 64, 32);
            geometry.rotateX(Math.PI / 2);
            geometry.scale(1, 1 / p.a_b, 1 / p.a_c);  // rapports d'axes SOCCA
            geometry.computeVertexNormals();

            const grayMaterial = new THREE.MeshLambertMaterial({ color: 0x777777, side: THREE.DoubleSide });
            const object = new THREE.Mesh(geometry, grayMaterial);

            // OBJ en memoire, puis liens View / Download
            const objUrl = URL.createObjectURL(new Blob([geometryToOBJ(geometry)], { type: "text/plain" }));

            const viewLink = document.getElementById("model3d-view-obj");
            viewLink.href = objUrl;
            viewLink.target = "_blank";

            const downloadLink = document.getElementById("model3d-download-obj");
            downloadLink.href = objUrl;
            downloadLink.download = "asteroid_socca.obj";

            // Centrage de l'objet
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const objectSize = box.getSize(new THREE.Vector3());
            const maxSize = Math.max(objectSize.x, objectSize.y, objectSize.z);
            object.position.sub(center);

            const asteroidGroup = new THREE.Group();
            asteroidGroup.add(object);

            // Orientation : SOCCA seule par defaut, corrigee periode ou DAMIT au clic
            let mode = "socca";   // "socca", "periode" ou "damit"

            // Un objet par correction : son bouton, son quaternion, son nom
            const corrections = {
                periode: {
                    button: document.getElementById("model3d-periode-button"),
                    quaternion: p.quaternion_periode,
                    info: "Period correction: adjusts the rotation phase using the DAMIT period."
                },
                damit: {
                    button: document.getElementById("model3d-damit-button"),
                    quaternion: p.quaternion_damit,
                    info: "DAMIT corrections: adjusts both the rotation period and reference longitude."
                }
            };

            function hasQuaternion(q) {
                return !!(q && q.length === 4);
            }

            function applyOrientation() {
                const c = corrections[mode];
                const q = (c && hasQuaternion(c.quaternion)) ? c.quaternion : p.quaternion;
                // ordre [x, y, z, w]
                if (q && q.length === 4) asteroidGroup.quaternion.set(q[0], q[1], q[2], q[3]).normalize();
            }

            // Met a jour les deux boutons : grise, actif, infobulle
            function updateButtons() {
                for (const key in corrections) {
                    const c = corrections[key];
                    const has = hasQuaternion(c.quaternion);
                    const on = mode === key && has;
                    c.button.disabled = !has;
                    c.button.classList.toggle("active", on);
                    c.button.title = !has ? "No model available"
                                   : on ? c.info + ": ON (click to disable)"
                                        : c.info + ": OFF (click to enable)";
                }
            }

            // Un clic : on active le mode, ou on revient a SOCCA s'il etait deja actif
            for (const key in corrections) {
                corrections[key].button.onclick = function () {
                    mode = (mode === key) ? "socca" : key;
                    updateButtons();
                    applyOrientation();
                    renderer.render(scene, camera);
                };
            }

            applyOrientation();
            scene.add(asteroidGroup);
            updateButtons();

            // Direction du Soleil
            const sunDirection = new THREE.Vector3(0, 0, 1);
            if (p.sun_direction) {
                sunDirection.set(p.sun_direction[0], p.sun_direction[1], p.sun_direction[2]);
            }
            sunDirection.normalize();
            const sunPosition = sunDirection.clone().multiplyScalar(maxSize * 10);
            addLights(scene, sunPosition);

            // Bouton globe : affiche / cache le quadrillage et les coordonnees
            const gridMaterial = new THREE.MeshLambertMaterial({ map: makeGridTexture(), side: THREE.DoubleSide });
            let gridVisible = false;
            const gridButton = document.getElementById("model3d-coordinates-button");

            gridButton.onclick = function () {
                gridVisible = !gridVisible;
                object.material = gridVisible ? gridMaterial : grayMaterial;
                gridButton.classList.toggle("active", gridVisible);
                renderer.render(scene, camera);
            };

            camera.position.set(0, 0, maxSize * zoom);
            camera.up.set(0, 1, 0);
            camera.near = maxSize * 0.01;
            camera.far = maxSize * 100;
            camera.updateProjectionMatrix();
            camera.lookAt(0, 0, 0);

            // Zoom
            function setZoom(factor) {
                zoom = Math.min(10, Math.max(1.5, zoom * factor));
                camera.position.set(0, 0, maxSize * zoom);
                renderer.render(scene, camera);
            }
            document.getElementById("model3d-zoom-in").onclick = function () { setZoom(0.8); };
            document.getElementById("model3d-zoom-out").onclick = function () { setZoom(1.25); };

             // rotation manuelle activee par le bouton
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.enableRotate = false;
            controls.enableZoom = false;
            controls.enablePan = false;

            const rotateButton = document.getElementById("model3d-rotate-button");
            rotateButton.onclick = function () {
                controls.enableRotate = !controls.enableRotate;
                rotateButton.title = controls.enableRotate ? "Disable manual rotation" : "Enable manual rotation";
                rotateButton.classList.toggle("active", controls.enableRotate);
            };

            // Mise à jour continue de la scène pour la rotation manuelle
            // Rotation automatique : un tour complet en 10 secondes
            let playing = false;
            let lastTime = null;
            const rotationDuration = 10;
            const playButton = document.getElementById("model3d-play-button");

            playButton.onclick = function () {
                playing = !playing;
                playButton.title = playing ? "Pause asteroid rotation" : "Play asteroid rotation";
                playButton.classList.toggle("active", playing);
                lastTime = null;
            };

            function animate(time) {
                requestAnimationFrame(animate);

                if (playing && lastTime !== null) {
                    asteroidGroup.rotateZ(2 * Math.PI * ((time - lastTime) / 1000) / rotationDuration);
                }
                lastTime = time;
                controls.update();
                renderer.render(scene, camera);
            }
            requestAnimationFrame(animate);

            return "";
        }
    }
});