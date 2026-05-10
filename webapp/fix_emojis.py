import os
import re

svgs = {
    '🔬': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 4.5L19.5 9.5M5 20L19 6M9 16L4 21M15 9L9 15"/></svg>',
    '🏥': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21h18M5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16M12 9v6M9 12h6"/></svg>',
    '🔍': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    '🧬': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    '🤖': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4M8 16h8"></path></svg>',
    '📋': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>',
    '🖼️': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>',
    '🖼': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>',
    '📊': '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="18" y="3" width="4" height="18"></rect><rect x="10" y="8" width="4" height="13"></rect><rect x="2" y="13" width="4" height="8"></rect></svg>',
    '🧠': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12c0 2.21.72 4.25 1.93 5.88L3 21l3.12-.93C7.75 21.28 9.79 22 12 22c5.52 22 10-17.52 10-12S17.52 2 12 2zm0 18c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6zm-1-9V7c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1s-1-.45-1-1zm0 4h2v2h-2v-2z"></path></svg>',
    '<span class="nav-logo-icon">🔬</span>': '<div class="nav-logo-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M14.5 4.5L19.5 9.5M5 20L19 6M9 16L4 21M15 9L9 15"/></svg></div>'
}

# The nav logo text in base template often has emoji, we should replace the nav bar appropriately!
files = ['doctor.html', 'index.html', 'screening.html', 'processing.html']
for fn in files:
    path = f'webapp/templates/{fn}'
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # The navbar logo text replace
    # <a href="/" class="logo">
    #   <span class="logo-icon">🔬</span>
    #   Multimodal UTI Prediction
    # </a>
    # The requirement is:
    # <a href="/" class="nav-logo">
    #   <div class="nav-logo-icon">
    #     <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M14.5 4.5L19.5 9.5M5 20L19 6M9 16L4 21M15 9L9 15"/></svg>
    #   </div>
    #   <span class="nav-logo-text">UTI <span>Screening</span></span>
    # </a>

    for emoji, svg in svgs.items():
        text = text.replace(emoji, svg)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

print("Done")