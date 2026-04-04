// @ts-check
const { expect } = require('@playwright/test');
const { BasePage } = require('./BasePage');

/**
 * Patterns page object model
 */
class PatternsPage extends BasePage {
  /**
   * @param {import('@playwright/test').Page} page
   */
  constructor(page) {
    super(page);
    
    // Selectors
    this.pageTitle = page.getByRole('heading', { name: /patterns/i });
    
    // Pattern cards
    this.patternCards = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /reflection|planning|tool|rAG|metacognition/i });
    this.reflectionPattern = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /reflection/i }).first();
    this.planningPattern = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /planning/i }).first();
    this.toolUsePattern = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /tool use/i }).first();
    this.agenticRAGPattern = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /agentic.*rag/i }).first();
    this.metacognitionPattern = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: /metacognition/i }).first();
    
    // Run button
    this.runButton = page.getByRole('button', { name: /run pattern|execute/i });
    
    // Visualization
    this.visualization = page.locator('[class*="visualization"], [class*="viz"], [class*="pattern-result"]');
    this.loadingState = page.locator('[class*="loading"], [class*="spinner"]').first();
    
    // Pattern details
    this.patternSteps = page.locator('[class*="step"]').first();
    this.patternOutput = page.locator('[class*="output"], [class*="result"]').first();
  }

  /**
   * Navigate to patterns page
   */
  async goto() {
    await this.navigate('/patterns');
    await this.waitForNetworkIdle();
  }

  /**
   * Verify patterns page is loaded
   */
  async expectLoaded() {
    await expect(this.pageTitle).toBeVisible();
  }

  /**
   * Select a pattern by name
   * @param {import('@playwright/test').Page} page
   * @param {string} patternName
   */
  async selectPattern(page, patternName) {
    const patternCard = page.locator('[class*="pattern"], [class*="card"]').filter({ hasText: patternName }).first();
    await patternCard.click();
  }

  /**
   * Run the selected pattern
   */
  async runPattern() {
    await this.runButton.click();
  }

  /**
   * Wait for pattern execution to complete
   */
  async waitForPatternComplete() {
    await expect(this.loadingState).not.toBeVisible({ timeout: 30000 });
    await expect(this.patternOutput).toBeVisible({ timeout: 30000 });
  }
}

module.exports = { PatternsPage };
