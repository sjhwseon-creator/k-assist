const routes = ["home", "about", "services", "process", "contact"];
const navLinks = document.querySelectorAll(".banner-nav a");
const pageSections = document.querySelectorAll(".page-section");

const getRouteFromHash = () => {
  const route = window.location.hash.replace("#", "");
  return routes.includes(route) ? route : "home";
};

const showPage = () => {
  const activeRoute = getRouteFromHash();

  pageSections.forEach((section) => {
    section.hidden = section.id !== activeRoute;
  });

  navLinks.forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${activeRoute}`);
  });

  window.scrollTo({ top: 0, behavior: "auto" });
};

showPage();
window.addEventListener("hashchange", showPage);
