// custom.js
document.addEventListener("DOMContentLoaded", function() {
    var sidebar = document.querySelector(".wy-nav-side");
    var content = document.querySelector(".wy-nav-content");
    var contentWrap = document.querySelector(".wy-nav-content-wrap");
    var toggleButton = document.createElement("button");
    toggleButton.className = "wy-nav-side-toggle";
    toggleButton.innerHTML = "☰";
    document.body.appendChild(toggleButton);

    toggleButton.addEventListener("click", function() {
        sidebar.classList.toggle("collapsed");
        content.classList.toggle("collapsed");
        contentWrap.classList.toggle("collapsed");
    });
});
