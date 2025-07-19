(function(){
  const listeners = {};
  let ws;
  let retry = 1000;
  const home = document.getElementById('homeIcon');
  function connect(){
    ws = new WebSocket(`ws://${location.hostname}:8766`);
    ws.onopen = () => {
      if(home) home.classList.add('connected');
      retry = 1000;
    };
    ws.onclose = () => {
      if(home) home.classList.remove('connected');
      retry = Math.min(retry * 1.5, 15000);
      setTimeout(connect, retry);
    };
    ws.onmessage = e => {
      let m;
      try { m = JSON.parse(e.data); } catch(err) { return; }
      if(m.view && listeners[m.view]){
        listeners[m.view].forEach(fn => fn(m));
      }
    };
  }
  connect();
  window.appEvents = {
    listen(view, fn){
      if(!listeners[view]) listeners[view] = [];
      listeners[view].push(fn);
    }
  };
})();
