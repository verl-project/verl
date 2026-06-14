// Resizable sidebar functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.querySelector('.wy-nav-side');
    const content = document.querySelector('.wy-nav-content-wrap');
    
    if (!sidebar || !content) return;
    
    // Create resize handle
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'resize-handle';
    sidebar.appendChild(resizeHandle);
    
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    // Get initial width
    const getInitialWidth = () => {
        return 300; // Default width
    };
    
    // Save width to localStorage
    const saveWidth = (width) => {
        localStorage.setItem('sidebar-width', width);
    };
    
    // Load width from localStorage
    const loadWidth = () => {
        const savedWidth = localStorage.getItem('sidebar-width');
        if (savedWidth) {
            const width = parseInt(savedWidth, 10);
            if (width >= 200 && width <= 600) {
                return width;
            }
        }
        return getInitialWidth();
    };
    
    // Apply width to sidebar and content
    const applyWidth = (width) => {
        // Update sidebar width
        sidebar.style.width = width + 'px';
        
        // Update content margin with !important to override any CSS
        content.style.setProperty('margin-left', width + 'px', 'important');
        
        // Also update any other content wrapper that might exist
        const contentInner = document.querySelector('.wy-nav-content');
        if (contentInner) {
            contentInner.style.setProperty('margin-left', '0px', 'important');
        }
        
        // Force reflow and repaint
        sidebar.offsetHeight;
        content.offsetHeight;
        
        // Trigger window resize event to notify other components
        window.dispatchEvent(new Event('resize'));
    };
    
    // Initialize with saved width
    const initialWidth = loadWidth();
    applyWidth(initialWidth);
    
    // Mouse down on resize handle
    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
        
        sidebar.classList.add('resizing');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        
        // Add overlay to prevent iframe issues
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 9999;
            cursor: col-resize;
        `;
        overlay.id = 'resize-overlay';
        document.body.appendChild(overlay);
        
        e.preventDefault();
    });
    
    // Mouse move
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const width = startWidth + e.clientX - startX;
        const clampedWidth = Math.max(200, Math.min(600, width));
        applyWidth(clampedWidth);
    });
    
    // Mouse up
    document.addEventListener('mouseup', () => {
        if (!isResizing) return;
        
        isResizing = false;
        sidebar.classList.remove('resizing');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        
        // Remove overlay
        const overlay = document.getElementById('resize-overlay');
        if (overlay) {
            overlay.remove();
        }
        
        // Save the current width
        const currentWidth = parseInt(window.getComputedStyle(sidebar).width, 10);
        saveWidth(currentWidth);
    });
    
    // Handle window resize - removed to prevent infinite loop
    // The sidebar width is fixed and managed by drag functionality, no need to recalculate on window resize
    
    // Double-click to reset to default width
    resizeHandle.addEventListener('dblclick', () => {
        const defaultWidth = 300;
        applyWidth(defaultWidth);
        saveWidth(defaultWidth);
    });
});

// Fix navigation issues - Using MutationObserver for reliable initialization
document.addEventListener('DOMContentLoaded', function() {
    let navigationFixed = false;
    
    function setupNavigationFix() {
        if (navigationFixed) return;
        
        // Find all links in the sidebar
        const sidebarLinks = document.querySelectorAll('.wy-menu-vertical a');
        
        // Only proceed if we have sidebar links
        if (sidebarLinks.length === 0) return;
        
        console.log('Setting up navigation fix...');
        
        sidebarLinks.forEach(function(link) {
            const href = link.getAttribute('href');
            
            // Clone the link to remove all existing event listeners
            const newLink = link.cloneNode(true);
            
            // Add our own click handler
            newLink.addEventListener('click', function(e) {
                console.log('Link clicked:', href);
                
                // If it's an anchor link within the same page
                if (href && href.startsWith('#') && href !== '#') {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const targetId = href.substring(1);
                    const targetElement = document.getElementById(targetId);
                    
                    if (targetElement) {
                        // Calculate offset for fixed header
                        const headerHeight = 60;
                        const elementPosition = targetElement.getBoundingClientRect().top;
                        const offsetPosition = elementPosition + window.pageYOffset - headerHeight;
                        
                        window.scrollTo({
                            top: offsetPosition,
                            behavior: 'smooth'
                        });
                        
                        // Update URL hash
                        if (history.pushState) {
                            history.pushState(null, null, '#' + targetId);
                        } else {
                            location.hash = '#' + targetId;
                        }
                    }
                }
                // For external links, navigate normally
                else if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                    console.log('Navigating to external link:', href);
                    window.location.href = href;
                }
            });
            
            // Replace the old link with the new one
            link.parentNode.replaceChild(newLink, link);
        });
        
        navigationFixed = true;
        
        // Handle initial page load with hash
        if (window.location.hash) {
            // Use requestAnimationFrame for better timing
            requestAnimationFrame(() => {
                const targetId = window.location.hash.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    const headerHeight = 60;
                    const elementPosition = targetElement.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerHeight;
                    
                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        }
    }
    
    // Try to set up navigation fix immediately
    setupNavigationFix();
    
    // If it didn't work, use MutationObserver to watch for when sidebar links are added
    if (!navigationFixed) {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Check if sidebar links were added
                    const sidebarLinks = document.querySelectorAll('.wy-menu-vertical a');
                    if (sidebarLinks.length > 0) {
                        setupNavigationFix();
                        if (navigationFixed) {
                            observer.disconnect();
                        }
                    }
                }
            });
        });
        
        // Start observing the document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Fallback timeout in case MutationObserver doesn't work
        setTimeout(function() {
            if (!navigationFixed) {
                setupNavigationFix();
            }
            observer.disconnect();
        }, 5000);
    }
});