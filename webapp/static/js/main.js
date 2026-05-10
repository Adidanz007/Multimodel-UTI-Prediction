// ============================================
//  UTI AI — Professional JavaScript
// ============================================

// === SMOOTH PAGE TRANSITIONS ===
document.addEventListener('DOMContentLoaded', () => {
  document.body.style.opacity = '0';
  document.body.style.transition = 'opacity 0.3s ease';
  requestAnimationFrame(() => {
    document.body.style.opacity = '1';
  });

  // Theme Toggle Logic
  const themeToggle = document.getElementById('themeToggle');
  const body = document.documentElement;
  
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        if (body.getAttribute('data-theme') === 'light') {
            body.removeAttribute('data-theme');
            localStorage.setItem('theme', 'dark');
        } else {
            body.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
        }
    });
  }

  // Add fade-out on navigation
  document.querySelectorAll('a:not([target="_blank"])').forEach(link => {
    if (link.href && !link.href.includes('#') && !link.href.includes('javascript')) {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        document.body.style.opacity = '0';
        setTimeout(() => { window.location.href = link.href; }, 280);
      });
    }
  });
});

// === MULTI-STEP FORM WIZARD ===
let currentStep = 1;
const totalSteps = 3;

function showStep(step) {
  document.querySelectorAll('.form-step').forEach((el, i) => {
    el.style.display = (i + 1 === step) ? 'block' : 'none';
    el.style.animation = (i + 1 === step) ? 'fadeInUp 0.4s ease forwards' : '';
  });
  document.querySelectorAll('.progress-step').forEach((el, i) => {
    el.classList.toggle('active', i + 1 === step);
    el.classList.toggle('completed', i + 1 < step);
  });
  const stepCounter = document.getElementById('stepCounter');
  if (stepCounter) stepCounter.textContent = `Step ${step} of ${totalSteps}`;
}

function nextStep() {
  if (validateCurrentStep() && currentStep < totalSteps) {
    currentStep++;
    showStep(currentStep);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function prevStep() {
  if (currentStep > 1) {
    currentStep--;
    showStep(currentStep);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

// === FORM VALIDATION ===
const FIELD_RANGES = {
  'age':              { min: 0,     max: 120,  unit: 'years' },
  'urine_ph':         { min: 4.5,   max: 8.5,  unit: '' },
  'specific_gravity': { min: 1.000, max: 1.035, unit: '' },
  'urine_wbc':        { min: 0,     max: 100,  unit: '/hpf' },
  'urine_rbc':        { min: 0,     max: 100,  unit: '/hpf' },
  'Temperature':      { min: 35.0,  max: 42.0, unit: '°C' },
  'RBC':              { min: 3.5,   max: 6.0,  unit: 'M/uL' },
  'WBC':              { min: 3.5,   max: 11.0, unit: 'K/uL' },
};

function validateCurrentStep() {
  let valid = true;
  const currentStepEl = document.querySelector(`.form-step:nth-child(${currentStep})`);
  if (!currentStepEl) return true;

  currentStepEl.querySelectorAll('.field-input[required]').forEach(input => {
    if (!input.value.trim()) {
      input.classList.remove('valid');
      input.classList.add('invalid');
      valid = false;
    } else {
      input.classList.remove('invalid');
      input.classList.add('valid');
    }
  });

  return valid;
}

function showFieldError(input, message) {
  let err = input.parentElement.querySelector('.field-error');
  if (!err) {
    err = document.createElement('div');
    err.className = 'field-error';
    err.style.cssText = 'font-size:0.72rem;color:var(--red);margin-top:4px;';
    input.parentElement.appendChild(err);
  }
  err.textContent = message;
}

// === IMAGE UPLOAD ===
function initImageUpload() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('imageInput');
  const preview = document.getElementById('imagePreview');
  if (!zone) return;

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleImageFile(input.files[0]);
  });

  function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = 'block';
      const placeholder = zone.querySelector('.upload-placeholder');
      if (placeholder) placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Show file info
    const info = document.getElementById('fileInfo');
    if (info) {
      info.textContent = `${file.name} — ${(file.size/1024).toFixed(1)} KB`;
      info.style.display = 'block';
    }
  }
}

// === FORM SUBMISSION ===
function submitScreening() {
  if (!validateCurrentStep()) return;

  const form = document.getElementById('screeningForm');
  if (!form) return;
  const formData = new FormData(form);

  const imageInput = document.getElementById('imageInput');
  if (!imageInput.files[0]) {
    alert('Please upload an ultrasound image');
    return;
  }

  formData.append('image', imageInput.files[0]);

  const submitBtn = document.querySelector('button[onclick="submitScreening()"]');
  const originalText = submitBtn ? submitBtn.innerHTML : 'Run AI Analysis';
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span style="display:inline-block; width:16px; height:16px; border:2px solid rgba(255,255,255,0.3); border-radius:50%; border-top-color:#fff; animation:spin 1s linear infinite; margin-right:8px; vertical-align:middle;"></span> Processing...';
    submitBtn.style.opacity = '0.8';
    submitBtn.style.cursor = 'wait';
  }

  fetch('/api/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        window.location.href = '/processing/' + data.result_id;
      } else {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML = originalText;
          submitBtn.style.opacity = '1';
          submitBtn.style.cursor = 'pointer';
        }
        alert('Error: ' + data.error);
      }
    })
    .catch(err => {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML = originalText;
          submitBtn.style.opacity = '1';
          submitBtn.style.cursor = 'pointer';
        }
        alert('Network error occurred.');
    });
}

// === PROCESSING PAGE ===
function initProcessing() {
  const steps = [
    { text: 'Receiving clinical data',        delay: 500  },
    { text: 'Analyzing biomarkers (XGBoost)', delay: 1200 },
    { text: 'Preprocessing ultrasound image', delay: 2000 },
    { text: 'Running EfficientNetB3',          delay: 2800 },
    { text: 'Multimodal fusion analysis',      delay: 3800 },
    { text: 'Generating Grad-CAM heatmap',     delay: 4800 },
    { text: 'Compiling report',               delay: 5500 },
  ];

  steps.forEach((step, i) => {
    setTimeout(() => {
      const el = document.getElementById(`step-${i}`);
      if (el) {
        el.classList.add('active');
        if (i > 0) {
          const prev = document.getElementById(`step-${i-1}`);
          if (prev) {
            prev.classList.remove('active');
            prev.classList.add('done');
          }
        }
      }
      updateProgressBar((i + 1) / steps.length * 100);
    }, step.delay);
  });

  // Poll for result
  function pollResult() {
    const result_id = sessionStorage.getItem('result_id');
    if (result_id) {
      window.location.href = `/result/${result_id}`;
    } else {
      setTimeout(pollResult, 500);
    }
  }

  setTimeout(pollResult, 3000);
}

function updateProgressBar(percent) {
  const bar = document.getElementById('progressBar');
  if (bar) bar.style.width = percent + '%';
}

// === RISK GAUGE ANIMATION ===
function animateGauge(targetPercent, color) {
  const canvas = document.getElementById('riskGauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = 200;
  canvas.height = 200;
  let current = 0;

  const interval = setInterval(() => {
    current = Math.min(current + 1.5, targetPercent);
    ctx.clearRect(0, 0, 200, 200);

    // Background track
    ctx.beginPath();
    ctx.arc(100, 100, 78, Math.PI * 0.75, Math.PI * 2.25);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Progress arc
    const startAngle = Math.PI * 0.75;
    const endAngle = startAngle + (current / 100) * Math.PI * 1.5;
    ctx.beginPath();
    ctx.arc(100, 100, 78, startAngle, endAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Glow effect
    ctx.beginPath();
    ctx.arc(100, 100, 78, startAngle, endAngle);
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.stroke();

    if (current >= targetPercent) clearInterval(interval);
  }, 16);
}

// === RISK TIMELINE ANIMATION ===
function animateTimeline(percent) {
  const marker = document.getElementById('timelineMarker');
  if (!marker) return;
  setTimeout(() => {
    marker.style.left = percent + '%';
  }, 300);
}

// === SCROLL ANIMATIONS ===
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.glass-card, .result-card, .rec-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(16px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });
}

// === CONFIDENCE PIPS ===
function renderConfidencePips(containerId, score, color) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const pips = 10;
  const filled = Math.round(score * pips);
  container.innerHTML = '';
  for (let i = 0; i < pips; i++) {
    const pip = document.createElement('div');
    pip.className = 'confidence-pip' + (i < filled ? ` active ${color}` : '');
    container.appendChild(pip);
  }
}

// === INIT ALL ===
document.addEventListener('DOMContentLoaded', () => {
  initScrollAnimations();
  initImageUpload();

  // If on result page
  const gauge = document.getElementById('riskGauge');
  if (gauge) {
    const score = parseFloat(gauge.dataset.score);
    const color = gauge.dataset.color || '#00D4BC';
    animateGauge(Math.round(score * 100), color);
    animateTimeline(Math.round(score * 100));
    renderConfidencePips('confidencePips', score, score > 0.65 ? 'red' : score > 0.48 ? 'amber' : 'teal');
  }

  // If on processing page
  if (document.getElementById('step-0')) initProcessing();

  // If on screening form
  if (document.getElementById('screeningForm')) showStep(1);
});