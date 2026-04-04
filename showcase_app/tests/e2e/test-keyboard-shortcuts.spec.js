// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Keyboard Shortcuts', () => {
  test('should show keyboard shortcuts modal with cmd+/', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Press Cmd+/
    await page.keyboard.press('Meta+/');
    
    // Modal should appear
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible({ timeout: 3000 });
    
    await page.screenshot({ path: 'test-results/keyboard-shortcuts-modal.png', fullPage: true });
  });

  test('should display shortcut information in modal', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open shortcuts modal
    await page.keyboard.press('Meta+/');
    
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible();
    
    // Should contain shortcut categories
    await expect(page.getByText(/navigation/i)).toBeVisible();
    await expect(page.getByText(/appearance/i)).toBeVisible();
    
    // Should list specific shortcuts
    await expect(page.getByText(/cmd\+b/i)).toBeVisible();
    await expect(page.getByText(/cmd\+d/i)).toBeVisible();
  });

  test('should close shortcuts modal with Escape', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open shortcuts modal
    await page.keyboard.press('Meta+/');
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible();
    
    // Press Escape
    await page.keyboard.press('Escape');
    
    // Modal should close
    await expect(modal).not.toBeVisible({ timeout: 2000 });
  });

  test('should toggle sidebar with cmd+b', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Sidebar should be visible initially
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
    
    // Get initial width or class
    const initialWidth = await sidebar.evaluate(el => el.className);
    
    // Press Cmd+B to toggle
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    // Sidebar should change (collapsed state)
    const newWidth = await sidebar.evaluate(el => el.className);
    
    // The class or width should be different
    expect(newWidth).not.toBe(initialWidth);
    
    await page.screenshot({ path: 'test-results/sidebar-toggled.png', fullPage: true });
  });

  test('should toggle theme with cmd+d', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Get initial theme
    const htmlElement = page.locator('html');
    const initialTheme = await htmlElement.evaluate(el => el.classList.contains('dark') ? 'dark' : 'light');
    
    // Press Cmd+D
    await page.keyboard.press('Meta+d');
    await page.waitForTimeout(500);
    
    // Theme should toggle
    const newTheme = await htmlElement.evaluate(el => el.classList.contains('dark') ? 'dark' : 'light');
    expect(newTheme).not.toBe(initialTheme);
  });

  test('should show shortcuts modal from sidebar button', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Click on Shortcuts button in sidebar
    const shortcutsButton = page.getByRole('button', { name: /shortcuts/i });
    await shortcutsButton.click();
    
    // Modal should appear
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible();
  });

  test('should close modal when clicking outside', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open shortcuts modal
    await page.keyboard.press('Meta+/');
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible();
    
    // Click outside modal
    await page.mouse.click(10, 10);
    
    // Modal should close
    await expect(modal).not.toBeVisible({ timeout: 2000 });
  });

  test('should display platform-specific shortcuts', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open shortcuts modal
    await page.keyboard.press('Meta+/');
    
    // Shortcuts should use Cmd for Mac or Ctrl for Windows/Linux
    // The display should adapt to the platform
    const modal = page.getByRole('dialog', { name: /keyboard shortcuts/i });
    await expect(modal).toBeVisible();
    
    // Should show recognizable shortcut patterns
    const modalContent = await modal.textContent();
    expect(modalContent.length).toBeGreaterThan(0);
  });
});
