/**
 * Make toctree caption sections (CONTENTS, EXAMPLES) collapsible in the sidebar.
 * 
 * The Read the Docs theme doesn't provide native support for collapsible caption
 * sections. This script adds toggle functionality so users can collapse/expand
 * these sections to manage sidebar space more efficiently.
 * 
 * Behavior:
 * - Captions start collapsed by default to reduce initial clutter
 * - EXCEPT: If current page is within a section, that section stays expanded
 * - Clicking a caption toggles the 'expanded' class
 * - CSS handles the visual states (arrow direction and ul visibility)
 */
document.addEventListener('DOMContentLoaded', function() {
    const captions = document.querySelectorAll('.wy-menu-vertical p.caption');
    
    captions.forEach(function(caption) {
        // Check if this caption's section contains the current page
        const nextUl = caption.nextElementSibling;
        const hasCurrentPage = nextUl && nextUl.querySelector('li.current');
        
        // Expand if section contains current page, otherwise collapse
        if (hasCurrentPage) {
            caption.classList.add('expanded');
        } else {
            caption.classList.remove('expanded');
        }
        
        // Add click handler to toggle expansion state
        caption.addEventListener('click', function() {
            this.classList.toggle('expanded');
        });
    });
});
