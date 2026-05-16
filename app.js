const navLinks = document.querySelectorAll(".banner-nav a");
const sections = [...document.querySelectorAll("main section[id]")];

const setActiveLink = () => {
  const scrollPosition = window.scrollY + window.innerHeight * 0.32;
  let activeId = "about";

  for (const section of sections) {
    if (section.offsetTop <= scrollPosition) {
      activeId = section.id;
    }
  }

  navLinks.forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${activeId}`);
  });
};

setActiveLink();
window.addEventListener("scroll", setActiveLink, { passive: true });
