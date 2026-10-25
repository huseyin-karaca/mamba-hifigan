/* The epoch slider on the Listen page.
 *
 * Each .mh-epochs block holds one range input and a set of <audio data-template="..."> elements whose src is
 * rebuilt as the slider moves. Only the sources the page ships exist, so the slider steps through those
 * epochs rather than every epoch; the list comes from data-epochs on the block.
 *
 * Audio that is playing keeps playing across a change, so you can hold the slider and hear the model improve.
 */
(function () {
  function wire(block) {
    if (block.dataset.wired === "1") return;
    block.dataset.wired = "1";

    var epochs = (block.dataset.epochs || "").split(",").map(function (s) {
      return s.trim();
    }).filter(Boolean);
    if (!epochs.length) return;

    var range = block.querySelector('input[type="range"]');
    var label = block.querySelector(".mh-epochs__value");
    var players = Array.prototype.slice.call(block.querySelectorAll("audio[data-template]"));

    range.min = 0;
    range.max = String(epochs.length - 1);
    range.step = "1";

    function apply(index) {
      var epoch = epochs[index];
      if (label) label.textContent = "epoch " + parseInt(epoch, 10);
      players.forEach(function (audio) {
        var next = audio.dataset.template.replace("{epoch}", epoch);
        if (audio.getAttribute("src") === next) return;
        var wasPlaying = !audio.paused && !audio.ended;
        audio.setAttribute("src", next);
        audio.load();
        if (wasPlaying) {
          var play = audio.play();
          if (play && play.catch) play.catch(function () {});
        }
      });
    }

    range.addEventListener("input", function () {
      apply(parseInt(range.value, 10));
    });
    apply(parseInt(range.value, 10) || 0);
  }

  function init() {
    document.querySelectorAll(".mh-epochs").forEach(wire);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
