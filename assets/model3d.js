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
    // Sommets
    for (let i = 0; i < positions.count; i++) {
        lines.push(`v ${positions.getX(i)} ${positions.getY(i)} ${positions.getZ(i)}`);
    }
    // Coordonnees de texture
    if (uvs) {
        for (let i = 0; i < uvs.count; i++) {
            lines.push(`vt ${uvs.getX(i)} ${uvs.getY(i)}`);
        }
    }
    // Normales
    if (normals) {
        for (let i = 0; i < normals.count; i++) {
            lines.push(`vn ${normals.getX(i)} ${normals.getY(i)} ${normals.getZ(i)}`);
        }
    }

    // les indices OBJ commencent a 1
    const vertexIndex = i => (indices ? indices.getX(i) : i) + 1;
    const count = indices ? indices.count : positions.count;
    // Faces triangulaires
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

// Axes X, Y, Z : fleches fines + petites lettres, par-dessus l'asteroide
function makeAxes(size) {
    const group = new THREE.Group();

    // Une fleche et une lettre par axe : [lettre, couleur, x, y, z]
    for (const [letter, color, x, y, z] of [["X", "#ff5555", 1, 0, 0], ["Y", "#55dd55", 0, 1, 0], ["Z", "#5599ff", 0, 0, 1]]) {
        const dir = new THREE.Vector3(x, y, z);

        // Fleche du centre vers l'exterieur (longueur, pointe, largeur de pointe)
        group.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), size, color, size * 0.12, size * 0.05));

        // Lettre dessinee sur un petit canvas 2D
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 64;
        const ctx = canvas.getContext("2d");
        ctx.font = "32px sans-serif";
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(letter, 32, 32);

        // Sprite : la lettre fait toujours face a la camera, placee juste apres la pointe
        const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas) }));
        label.position.copy(dir).multiplyScalar(size * 1.15);
        label.scale.setScalar(size * 0.2);
        group.add(label);
    }

    // Par-dessus l'asteroide, legerement transparent pour rester discret
    // (traverse : applique le reglage a la ligne et au cone de chaque fleche, et aux lettres)
    group.traverse(o => {
        o.renderOrder = 10;               // dessine apres l'asteroide
        if (o.material) {
            o.material.depthTest = false; // visible meme a travers l'asteroide
            o.material.transparent = true;
            o.material.opacity = 0.85;
        }
    });

    return group;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    model3d: {
        draw: function (p) {
            if (!p || !p.a_b || !p.a_c) return "";

            const container = document.getElementById("model-3d");
            if (!container) return "";

            // Supprime l'ancien rendu
            if (window.model3dCleanup) window.model3dCleanup();
            container.innerHTML = "";
            // Etat initial des boutons (draw peut etre relance par le Reset)
            document.querySelectorAll(".model3d-view button").forEach(b => b.classList.remove("active"));
            document.getElementById("model3d-play-button").title = "Play / Pause asteroid rotation";
            document.getElementById("model3d-rotate-button").title = "Enable manual rotation";

            const resetButton = document.getElementById("model3d-reset-button");
            resetButton.disabled = true;   // grise tant que rien n'a change
            // Creation de la scene, de la camera et du rendu
            const w = container.clientWidth, h = container.clientHeight;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);

            const camera = new THREE.PerspectiveCamera(25, w / h, 0.1, 1000);
            let zoom = 4;
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(w, h);
            container.appendChild(renderer.domElement);

            // Rendu a la demande
            let rafId = null;          // image en attente 
            let lastTime = null;       // temps de la derniere image
            let playing = false;       // rotation automatique active ou non
            let visible = true;        // false quand le conteneur n'est plus affiche
            const rotationDuration = 10;   // secondes par tour

            // Demande une nouvelle image
            function requestRender() {
                if (!rafId) rafId = requestAnimationFrame(frame);
            }

            function frame(time) {
                rafId = null;   // image creee

                if (!visible) {      // conteneur cache : on arrete la boucle
                    lastTime = null;
                    return;
                }

                if (playing) {          // Rotation de l'asteroide si la lecture est active
                    if (lastTime !== null) {
                        // plafonne : evite un grand saut apres un onglet cache
                        const dt = Math.min(time - lastTime, 100);   // Rotation autour de l'axe Z
                        asteroidGroup.rotateZ(2 * Math.PI * dt / (rotationDuration * 1000));
                    }
                    lastTime = time;
                } else {
                    lastTime = null;
                }

                const moved = controls.update();   // true tant que la camera bouge
                renderer.render(scene, camera);

                if (playing || moved) requestRender();   // on continue seulement si besoin
            }

            // Pause du rendu quand le conteneur n'est plus visible (changement d'onglet Dash)
            const visibilityObserver = new IntersectionObserver(function (entries) {
                visible = entries[0].isIntersecting;
                if (visible) requestRender();   // au retour, on redessine
            });
            visibilityObserver.observe(container);

            // Plein ecran + adaptation du rendu a la taille du conteneur
            const view = document.querySelector(".model3d-view");
            document.getElementById("model3d-fullscreen").onclick = function () {
                if (!document.fullscreenElement) view.requestFullscreen();
                else document.exitFullscreen();
            };
            const onFullscreen = function () {
                renderer.setSize(container.clientWidth, container.clientHeight);
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                requestRender();
            };
            document.addEventListener("fullscreenchange", onFullscreen);

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

            // Axes du repere de l'asteroide
            const axes = makeAxes(maxSize * 0.6);
            asteroidGroup.add(axes);

            // Orientation SOCCA par defaut.
            // Les modes P (periode) et D (periode + longitude de reference) servent uniquement a comparer les orientations avec DAMIT.
            // Ces boutons sont temporaires et seront retires de la version finale.
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

            function hasQuaternion(q) {     // Verifie si le quaternion est disponible
                return !!(q && q.length === 4);
            }

            function applyOrientation() { // Applique l'orientation choisie
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
                    requestRender();
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
                requestRender();
            };

            // Camera
            camera.position.set(0, 0, maxSize * zoom);
            camera.up.set(0, 1, 0);
            camera.near = maxSize * 0.01;
            camera.far = maxSize * 100;
            camera.updateProjectionMatrix();
            camera.lookAt(0, 0, 0);

            // Zoom
            function setZoom(factor) {
                zoom = Math.min(10, Math.max(1.5, zoom * factor));
                camera.position.setLength(maxSize * zoom);   // change la distance, garde l'angle de vue
                requestRender();
            }
            document.getElementById("model3d-zoom-in").onclick = function () { setZoom(0.8); };
            document.getElementById("model3d-zoom-out").onclick = function () { setZoom(1.25); };

            // Rotation manuelle (souris), activee par le bouton
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.enableRotate = controls.enableZoom = controls.enablePan = false;
            controls.addEventListener("change", requestRender);   // souris + inertie

            // Reset : on relance simplement le dessin
            resetButton.onclick = function () { window.dash_clientside.model3d.draw(p); };

            // Le bouton s'active des qu'on clique sur un bouton ou qu'on bouge la souris
            view.onclick = function (e) {
                const b = e.target.closest("button");
                if (b && b !== resetButton) resetButton.disabled = false;
            };
            controls.addEventListener("change", function () { resetButton.disabled = false; });

            const rotateButton = document.getElementById("model3d-rotate-button");
            rotateButton.onclick = function () {
                controls.enableRotate = !controls.enableRotate;
                rotateButton.title = controls.enableRotate ? "Disable manual rotation" : "Enable manual rotation";
                rotateButton.classList.toggle("active", controls.enableRotate);
            };

            // Rotation automatique : un tour complet en 10 secondes
            const playButton = document.getElementById("model3d-play-button");
            playButton.onclick = function () {
                playing = !playing;
                playButton.title = playing ? "Pause asteroid rotation" : "Play asteroid rotation";
                playButton.classList.toggle("active", playing);
                requestRender();   // relance la boucle (eteinte en pause)
            };
            // Afficher ou cacher les axes
            const axesButton = document.getElementById("model3d-axes-button");
            axesButton.classList.add("active");   // les axes sont visibles au depart
            axesButton.onclick = function () {
                axes.visible = !axes.visible;
                axesButton.classList.toggle("active", axes.visible);
                requestRender();
            };
            

            // Libere les ressources de l'ancien rendu
            window.model3dCleanup = function () {
                if (rafId) cancelAnimationFrame(rafId);   // annule l'image en attente
                visibilityObserver.disconnect();          // arrete la surveillance du conteneur
                document.removeEventListener("fullscreenchange", onFullscreen); // retire l'ecoute du plein ecran
                controls.dispose();                       // supprime les controles de la camera
                geometry.dispose();                       // libere la geometrie
                grayMaterial.dispose();                   // libere le materiau gris
                gridMaterial.map.dispose();               // libere la texture du quadrillage
                gridMaterial.dispose();                   // libere le materiau du quadrillage
                renderer.dispose();                       // libere les ressources du rendu
                renderer.forceContextLoss();              // libere le contexte WebGL
                URL.revokeObjectURL(objUrl);              // supprime l'URL temporaire du fichier OBJ
            };

            requestRender();   // premiere image

            return "";
        }
    }
});