// @ts-check
const { test, expect } = require('@playwright/test');
const { BasePage } = require('./BasePage');

/**
 * Home page object model
 */
class HomePage extends BasePage {
  /**
   * @param {import('@playwright/test').Page} page
   */
  constructor(page) {
    super(page);
    
    // Selectors
    this.navbar = page.getByRole('navigation');
    this.homeLink = page.getByRole('link', { name: /home/i });
    this.pipelinesLink = page.getByRole('link', { name: /pipelines/i });
    this.metricsLink = page.getByRole('link', { name: /metrics/i });
    this.patternsLink = page.getByRole('link', { name: /patterns/i });
    
    // Hero section
    this.heroSection = page.getByRole('heading', { name: /ia modules showcase/i });
    this.quickStartButton = page.getByRole('button', { name: /quick start|get started/i });
    
    // Feature cards
    this.featureCards = page.locator('[class*="card"], [class*="feature"]');
  }

  /**
   * Navigate to home page
   */
  async goto() {
    await this.navigate('/');
    await this.waitForNetworkIdle();
  }

  /**
   * Verify home page is loaded
   */
  async expectLoaded() {
    await expect(this.heroSection).toBeVisible();
  }
}

module.exports = { HomePage };
