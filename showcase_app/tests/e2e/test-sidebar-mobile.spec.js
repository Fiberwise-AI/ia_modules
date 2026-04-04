// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Sidebar & Mobile Menu', () => {
  test('should display sidebar by default', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
  });

  test('should toggle sidebar with keyboard shortcut', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    const sidebar = page.locator('aside');
    const initialClass = await sidebar.getAttribute('class');
    
    // Press Cmd+B to toggle
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    const newClass = await sidebar.getAttribute('class');
    expect(newClass).not.toBe(initialClass);
  });

  test('should collapse sidebar to icon-only mode', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Toggle to collapse
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    // Sidebar should be narrower (check for specific width class or style)
    const sidebar = page.locator('aside');
    const sidebarClass = await sidebar.getAttribute('class');
    
    // Should contain width indicator for collapsed state
    expect(sidebarClass).toMatch(/w-20|collapsed|w-16/);
    
    await page.screenshot({ path: 'test-results/sidebar-collapsed.png', fullPage: true });
  });

  test('should expand sidebar from collapsed state', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Collapse first
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    // Expand again
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    // Sidebar should show full width
    const sidebar = page.locator('aside');
    const sidebarClass = await sidebar.getAttribute('class');
    
    expect(sidebarClass).toMatch(/w-64|w-72/);
  });

  test('should show mobile menu button on small screens', async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 }
    });
    const page = await context.newPage();
    
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Mobile menu button should be visible
    const mobileMenuButton = page.getByRole('button').first();
    await expect(mobileMenuButton).toBeVisible();
    
    await page.close();
    await context.close();
  });

  test('should open mobile menu', async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 }
    });
    const page = await context.newPage();
    
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Click mobile menu button
    const mobileMenuButton = page.getByRole('button').first();
    await mobileMenuButton.click();
    
    // Sidebar should slide in
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
    
    await page.screenshot({ path: 'test-results/mobile-menu-open.png', fullPage: true });
    
    await page.close();
    await context.close();
  });

  test('should close mobile menu with backdrop click', async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 }
    });
    const page = await context.newPage();
    
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open mobile menu
    const mobileMenuButton = page.getByRole('button').first();
    await mobileMenuButton.click();
    
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
    
    // Click backdrop
    const backdrop = page.locator('[class*="backdrop"], [class*="overlay"]').first();
    if (await backdrop.isVisible()) {
      await backdrop.click();
      
      // Sidebar should hide
      await expect(sidebar).not.toBeVisible();
    }
    
    await page.close();
    await context.close();
  });

  test('should close mobile menu with Escape key', async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 }
    });
    const page = await context.newPage();
    
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open mobile menu
    const mobileMenuButton = page.getByRole('button').first();
    await mobileMenuButton.click();
    
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
    
    // Press Escape
    await page.keyboard.press('Escape');
    
    // Sidebar should hide
    await expect(sidebar).not.toBeVisible();
    
    await page.close();
    await context.close();
  });

  test('should navigate from mobile menu', async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 }
    });
    const page = await context.newPage();
    
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Open mobile menu
    const mobileMenuButton = page.getByRole('button').first();
    await mobileMenuButton.click();
    
    // Click pipelines link
    const pipelinesLink = page.getByRole('link', { name: /pipelines/i });
    await pipelinesLink.click();
    
    // Should navigate and close menu
    await page.waitForURL('**/pipelines');
    await expect(page.getByRole('heading', { name: /pipelines/i })).toBeVisible();
    
    await page.close();
    await context.close();
  });

  test('should highlight active navigation link', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Home link should be active
    const homeLink = page.locator('nav a[href="/"]').first();
    const homeClass = await homeLink.getAttribute('class');
    expect(homeClass).toMatch(/active|bg-primary|current/);
    
    // Navigate to pipelines
    await page.getByRole('link', { name: /pipelines/i }).click();
    await page.waitForURL('**/pipelines');
    
    // Pipelines link should now be active
    const pipelinesLink = page.locator('nav a[href="/pipelines"]').first();
    const pipelinesClass = await pipelinesLink.getAttribute('class');
    expect(pipelinesClass).toMatch(/active|bg-primary|current/);
  });

  test('should display version in sidebar', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    await expect(page.getByText(/showcase v/i)).toBeVisible();
  });

  test('should maintain sidebar state across navigation', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Collapse sidebar
    await page.keyboard.press('Meta+b');
    await page.waitForTimeout(500);
    
    const sidebar = page.locator('aside');
    const collapsedClass = await sidebar.getAttribute('class');
    
    // Navigate
    await page.getByRole('link', { name: /pipelines/i }).click();
    await page.waitForURL('**/pipelines');
    
    // Sidebar should still be collapsed
    const newClass = await sidebar.getAttribute('class');
    expect(newClass).toBe(collapsedClass);
  });
});
