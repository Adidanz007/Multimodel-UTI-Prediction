/* ===============================
NAVBAR SHADOW ON SCROLL
================================ */

window.addEventListener("scroll", function () {

    const navbar = document.querySelector(".navbar");

    if (window.scrollY > 20) {

        navbar.style.boxShadow = "0 10px 25px rgba(0,0,0,0.08)";

    } else {

        navbar.style.boxShadow = "0 4px 15px rgba(0,0,0,0.06)";
    }
});


/* ===============================
SCROLL REVEAL ANIMATION
================================ */

const observer = new IntersectionObserver(entries => {

    entries.forEach(entry => {

        if (entry.isIntersecting) {

            entry.target.classList.add("show");

        }

    });

});

const hiddenElements = document.querySelectorAll(".card, .feature, .hero h1, .hero p");

hiddenElements.forEach(el => observer.observe(el));



/* ===============================
BUTTON RIPPLE EFFECT
================================ */

const buttons = document.querySelectorAll(".primary-btn, .secondary-btn");

buttons.forEach(btn => {

    btn.addEventListener("click", function (e) {

        const ripple = document.createElement("span");

        const rect = btn.getBoundingClientRect();

        const size = Math.max(rect.width, rect.height);

        ripple.style.width = ripple.style.height = size + "px";

        ripple.style.left = e.clientX - rect.left - size / 2 + "px";

        ripple.style.top = e.clientY - rect.top - size / 2 + "px";

        ripple.classList.add("ripple");

        btn.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);

    });

});



/* ===============================
CARD HOVER 3D EFFECT
================================ */

const cards = document.querySelectorAll(".card");

cards.forEach(card => {

    card.addEventListener("mousemove", (e) => {

        const rect = card.getBoundingClientRect();

        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = -(y - centerY) / 15;
        const rotateY = (x - centerX) / 15;

        card.style.transform =
            `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.03)`;

    });

    card.addEventListener("mouseleave", () => {

        card.style.transform = "rotateX(0) rotateY(0) scale(1)";

    });

});



/* ===============================
SMOOTH SCROLL FOR LINKS
================================ */

document.querySelectorAll("a[href^='#']").forEach(anchor => {

    anchor.addEventListener("click", function (e) {

        e.preventDefault();

        const target = document.querySelector(this.getAttribute("href"));

        if (target) {

            target.scrollIntoView({
                behavior: "smooth"
            });

        }

    });

});