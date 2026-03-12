// Run after page loads

document.addEventListener("DOMContentLoaded", function () {

    const reveals = document.querySelectorAll(".reveal");

    function revealOnScroll() {

        const windowHeight = window.innerHeight;

        reveals.forEach(function (element) {

            const elementTop = element.getBoundingClientRect().top;

            const revealPoint = 100;

            if (elementTop < windowHeight - revealPoint) {

                element.classList.add("active");

            }

        });

    }

    window.addEventListener("scroll", revealOnScroll);

    revealOnScroll();

});



/* Navbar shadow on scroll */

window.addEventListener("scroll", function () {

    const navbar = document.querySelector(".navbar");

    if (window.scrollY > 20) {

        navbar.style.boxShadow = "0 8px 25px rgba(0,0,0,0.1)";

    } else {

        navbar.style.boxShadow = "0 4px 15px rgba(0,0,0,0.05)";
    }

});