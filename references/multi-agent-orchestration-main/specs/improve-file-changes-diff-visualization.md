# Plan: Improve Git Diff Visualization in FileChangesDisplay Component

## Task Description
Enhance the git diff review experience in the `FileChangesDisplay.vue` component by adding proper color coding (green for additions, red for deletions) and improving scannability. Currently, diffs are displayed as plain monochrome text making it difficult to quickly identify changes. This task focuses solely on scoped UI improvements and theme adjustments within the component.

## Objective
Transform the diff viewer from plain text to a properly styled, color-coded display that follows standard git diff conventions:
- Green background/text for added lines (+ prefix)
- Red background/text for deleted lines (- prefix)
- Subtle styling for diff headers (@@ markers)
- Improved line spacing and readability
- Maintain dark theme consistency with existing design

## Problem Statement
The current diff viewer in `FileChangesDisplay.vue` (lines 88-90, 169-171) renders diffs as plain text using `<pre><code class="language-diff">`. While it attempts to use highlight.js for syntax highlighting, the output lacks proper visual distinction between additions and deletions, making code reviews tedious. Users must mentally parse the diff symbols (+/-) rather than instantly recognizing changes through color coding.

## Solution Approach
Implement CSS-only diff syntax highlighting using attribute selectors and pseudo-elements to style diff lines based on their prefix characters. This approach:
1. Eliminates dependency on external highlighting libraries
2. Provides instant, consistent rendering
3. Maintains component encapsulation with scoped styles
4. Follows conventional git diff color schemes (green/red)
5. Integrates seamlessly with existing dark theme variables

## Relevant Files

### Existing Files to Modify
- **apps/orchestrator_3_stream/frontend/src/components/event-rows/FileChangesDisplay.vue** - Main component containing the diff viewer. Will add enhanced CSS styling for diff syntax highlighting in the scoped styles section (lines 301-670). No template or script changes needed.

### Reference Files
- **apps/orchestrator_3_stream/frontend/src/styles/global.css** - Global theme variables for color consistency. Will reference existing CSS variables like `--status-success`, `--status-error`, `--text-muted` for diff styling.
- **apps/orchestrator_3_stream/frontend/src/types.d.ts** - FileChange interface definition showing `diff?: string` field. No changes needed, but confirms diff content structure.

## Step by Step Tasks

### 1. Analyze Current Diff Rendering
- Review lines 88-90 and 169-171 in FileChangesDisplay.vue to understand current diff display structure
- Examine lines 587-605 (`.diff-viewer` styles) to understand existing styling baseline
- Verify the diff content format from `file.diff` property (raw git diff text with +/- prefixes)
- Note highlight.js integration attempt (lines 291-298) which will be superseded by CSS styling

### 2. Design Diff Line Styling Rules
- Define color scheme using existing theme variables:
  - Additions: `--status-success` (#10b981) with rgba background
  - Deletions: `--status-error` (#ef4444) with rgba background
  - Headers (@@ markers): `--status-info` or `--text-muted`
  - Context lines: default `--text-primary` color
- Plan line-by-line styling approach using CSS that targets line content
- Ensure sufficient contrast against dark background for accessibility

### 3. Implement CSS Diff Syntax Highlighting
- Add comprehensive CSS rules to `.diff-viewer` section (after line 605)
- Style lines based on git diff prefixes:
  - Lines starting with `+` (additions) → green background + text
  - Lines starting with `-` (deletions) → red background + text
  - Lines starting with `@@` (chunk headers) → muted blue/gray styling
  - Lines starting with `+++` or `---` (file headers) → dimmed styling
  - Context lines (no prefix) → default styling
- Add proper padding and line-height for improved scannability
- Use `white-space: pre-wrap` to handle long lines gracefully

### 4. Enhance Diff Viewer Container Styling
- Improve `.diff-viewer` container styles (lines 587-594):
  - Adjust `max-height` if needed for better initial view
  - Ensure scrollbar styling is consistent with theme
  - Add subtle border or shadow for better visual containment
- Enhance `.diff-viewer pre` and `.diff-viewer code` styles (lines 596-605):
  - Ensure proper line wrapping behavior
  - Add subtle line number visual guides if beneficial
  - Maintain monospace font consistency

### 5. Add Hover and Focus States
- Implement subtle hover effects on individual diff lines for better interaction feedback
- Consider adding line selection styling for copying specific lines
- Ensure hover states don't interfere with text selection

### 6. Test Color Contrast and Readability
- Verify green/red colors are distinguishable for colorblind users (use sufficient brightness difference)
- Test readability with various diff sizes (small changes vs large refactors)
- Ensure text remains readable when selected (selection highlighting)
- Validate against WCAG contrast guidelines

### 7. Validate with Browser Dev Tools
- Open the component in browser with sample diff data
- Inspect diff viewer styles in browser DevTools
- Verify CSS selectors are working correctly for all diff line types
- Check for any CSS specificity conflicts with global styles
- Test with both layout modes (default and two-column)

### 8. Cross-browser Compatibility Check
- Test in Chrome/Edge (primary browser)
- Verify in Firefox if available
- Check Safari rendering if on macOS
- Ensure no CSS features are used that lack broad support

### 9. Performance Verification
- Test with large diffs (500+ lines) to ensure no rendering lag
- Verify smooth scrolling within `.diff-viewer` container
- Check that scoped styles don't cause excessive CSS recalculation

## Testing Strategy

### Visual Testing
1. Create or use existing file changes with various diff types:
   - Small changes (1-5 lines)
   - Medium changes (20-50 lines)
   - Large refactors (100+ lines)
   - Mixed additions and deletions
   - Whole file creations/deletions

2. Verify color coding appears correctly:
   - Green backgrounds for `+` lines
   - Red backgrounds for `-` lines
   - Muted colors for `@@` headers
   - Proper text contrast in all cases

3. Test readability:
   - Quickly scan and identify change locations
   - Verify improved visual hierarchy vs original
   - Ensure no eye strain with new color scheme

### Integration Testing
- Expand/collapse diff sections to verify state handling
- Test in both "default" and "two-column" layout modes
- Verify no style conflicts with parent component styles
- Check file-open-in-IDE functionality still works

### Accessibility Testing
- Test with browser zoom at 150% and 200%
- Verify keyboard navigation through diff viewer
- Check screen reader compatibility (aria labels intact)
- Validate color contrast ratios meet WCAG AA standards

## Acceptance Criteria

- [ ] Added lines (starting with `+`) display with green background/text color
- [ ] Deleted lines (starting with `-`) display with red background/text color
- [ ] Diff chunk headers (starting with `@@`) display with muted styling
- [ ] File headers (`+++`, `---`) display with dimmed styling
- [ ] Context lines remain readable with default text color
- [ ] All diff styling uses scoped CSS within FileChangesDisplay.vue
- [ ] Color scheme integrates with existing theme variables from global.css
- [ ] Diff viewer remains responsive and handles long lines appropriately
- [ ] No performance degradation with large diffs (tested up to 500+ lines)
- [ ] Styles work correctly in both "default" and "two-column" layout modes
- [ ] Hover states provide clear interaction feedback without interfering with selection
- [ ] Text selection and copying works properly with new styling
- [ ] Color contrast meets WCAG AA accessibility standards
- [ ] No visual regressions in other parts of FileChangesDisplay component

## Validation Commands

Execute these commands to validate the task is complete:

```bash
# 1. Verify FileChangesDisplay.vue was modified with diff styling
git diff apps/orchestrator_3_stream/frontend/src/components/event-rows/FileChangesDisplay.vue

# 2. Check CSS syntax is valid (no linting errors)
cd apps/orchestrator_3_stream/frontend
npm run lint

# 3. Build the frontend to ensure no compilation errors
npm run build

# 4. Start the development server and visually verify
npm run dev
# Then open browser to http://localhost:5175 and navigate to view with file changes
```

## Notes

### Design Decisions
- **CSS-only approach**: Removes dependency on highlight.js which may not load consistently
- **Scoped styles**: Keeps changes isolated to this component, avoiding global style pollution
- **Theme variable reuse**: Maintains design system consistency by using existing color variables
- **Line-based styling**: Uses CSS pseudo-selectors to target individual lines within the diff for precise control

### Color Palette
Based on existing theme variables in global.css:
- **Additions**: `--status-success: #10b981` (green) with `rgba(16, 185, 129, 0.15)` background
- **Deletions**: `--status-error: #ef4444` (red) with `rgba(239, 68, 68, 0.15)` background
- **Headers**: `--status-info: #3b82f6` (blue) or `--text-muted: #6b7280` (gray)
- **Context**: `--text-primary: #ffffff` (default white text)

### Alternative Approaches Considered
1. **highlight.js**: Current approach but unreliable, adds dependency, increases bundle size
2. **Vue directive for line parsing**: Overcomplicated, requires script changes, harder to maintain
3. **Split diff into Vue components per line**: Performance overhead, unnecessary complexity
4. **Third-party diff viewer library**: Adds dependency, style customization difficult, bundle size impact

### Future Enhancements (Out of Scope)
- Line numbers column for easier reference
- Inline diff view (word-level changes highlighted)
- Copy individual line functionality
- Expand/collapse unchanged sections
- Side-by-side diff view option
- Syntax highlighting within diff lines (language-aware)

These enhancements would require script/template changes and are beyond the "simple scoped UI changes" constraint of this task.
