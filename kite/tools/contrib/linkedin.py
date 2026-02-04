from typing import Dict, List, Optional
import asyncio
import os
import random
import urllib.parse
from linkedin_scraper import BrowserManager

async def is_really_logged_in(page):
    """A bulletproof check for LinkedIn login state using multiple signals."""
    url = page.url.lower()
    
    # Signal 1: URL path
    logged_in_paths = ["/feed", "/mynetwork", "/messaging", "/notifications", "/me", "/company"]
    if any(path in url for path in logged_in_paths):
        if "/login" not in url and "/signup" not in url and "authwall" not in url:
            return True
    
    # Signal 2: Common modern UI elements
    selectors = [
        '.global-nav', '#global-nav', '.nav-item', 
        'button.global-nav__primary-link', 'img.global-nav__me-photo',
        '.feed-identity-module', '.share-box-feed-entry__trigger'
    ]
    
    for selector in selectors:
        try:
            if await page.locator(selector).count() > 0:
                return True
        except:
            continue
            
    return False

async def create_linkedin_session(session_path: str = "linkedin_session.json", **kwargs):
    """Create or renew a LinkedIn session file with smart detection."""
    print("="*60)
    print("LinkedIn Session Manager (Smart Renewal)")
    print("="*60)
    
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # Use headless=False so the user can see and interact
    async with BrowserManager(headless=False, user_agent=ua) as browser:
        # 1. Attempt to load existing session
        if os.path.exists(session_path):
            print(f"📂 Found existing session: {session_path}. Loading...")
            await browser.load_session(session_path)
            await browser.page.goto("https://www.linkedin.com/feed/")
            await asyncio.sleep(4)
        else:
            print("No existing session found. Starting fresh...")
            await browser.page.goto("https://www.linkedin.com/login")
            await asyncio.sleep(2)
        
        # 2. Check if already logged in
        if await is_really_logged_in(browser.page):
            print("\n✅ SUCCESS: You are already logged in via valid session.")
        else:
            print("\n🔐 Session invalid or missing. Please log in manually in the opened browser.")
            print("   I will auto-detect once you are on the Feed or any main page.")
        
        # 3. Wait for login / Detection
        print("\n⏳ Monitoring login status (timeout 5 mins)...\n")
        start_time = asyncio.get_event_loop().time()
        logged_in = False
        
        while (asyncio.get_event_loop().time() - start_time) < 300: # 5 minutes
            if await is_really_logged_in(browser.page):
                logged_in = True
                break
            await asyncio.sleep(2)
            
        if logged_in:
            print(f"\n✅ VERIFIED: Login state confirmed!")
            print(f"💾 Saving/Updating session to {session_path}...")
            await asyncio.sleep(3)
            # Ensure we are on a safe page before saving to avoid capturing transient states
            await browser.save_session(session_path)
            print("\n✅ DONE! Session is now current.")
            return True
        else:
            print("\n❌ FAILED: Login not detected within 5 minutes.")
            return False

def _resolve_session_path(provided_path: str) -> Optional[str]:
    """Robust resolution for session file in multiple locations."""
    # 1. Project Root (Preferred)
    cwd = os.getcwd()
    root_path = os.path.join(cwd, "linkedin_session_v2.json")
    if os.path.exists(root_path): return root_path
    if os.path.exists(provided_path): return provided_path
    
    fallbacks = [
        "linkedin_session_v2.json", 
        "linkedin_session.json", 
        os.path.join(cwd, "linkedin_session.json")
    ]
    for f in fallbacks:
        if os.path.exists(f): return f
    return None

async def search_linkedin_posts(query: str, limit: int = 30, session_path: str = "linkedin_session.json", **kwargs) -> List[Dict]:
    """Search for LinkedIn posts."""
    fw = kwargs.get('framework')
    def tool_log(msg):
        if fw: fw.event_bus.emit("tool:log", {"agent": "LinkedInTool", "message": msg})
        else: print(f"   [Tool] {msg}")

    tool_log(f"Starting search for: {query} (limit={limit})")
    print(f"\n📡 Running LinkedIn search for: '{query}' (Target: {limit} posts)")
    
    if "keywords" in kwargs:
        if isinstance(kwargs["keywords"], list): query += " " + " ".join(kwargs["keywords"])
        elif isinstance(kwargs["keywords"], str): query += " " + kwargs["keywords"]
    
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    results = []
    all_content = set()
    
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        active_session = _resolve_session_path(session_path)
        if active_session:
            tool_log(f"Loading session: {active_session}")
            await bm.load_session(active_session)
        else:
            tool_log("CRITICAL: No session file found!")
            return [{"error": "SESSION_MISSING", "message": "LinkedIn session file not found. Please run 'python3 create_new_session.py'."}]

        encoded_query = urllib.parse.quote(query)
        date_posted = kwargs.get("date_posted", "past-week")
        search_url = f"https://www.linkedin.com/search/results/content/?keywords={encoded_query}&origin=GLOBAL_SEARCH_HEADER"
        
        if date_posted:
            # Common LinkedIn date filters: "past-24h", "past-week", "past-month"
            search_url += f"&datePosted=%5B%22{date_posted}%22%5D"
        
        tool_log(f"Navigating to: {search_url}")
        await bm.page.goto(search_url, wait_until="domcontentloaded", timeout=90000)
        await asyncio.sleep(5)
        
        try:
            posts_filter = bm.page.locator('button:has-text("Posts")').first
            if await posts_filter.is_visible():
                await posts_filter.click()
                tool_log("Applied 'Posts' filter.")
                await asyncio.sleep(3)
        except: pass
        
        # Initial wait for rehydration
        await asyncio.sleep(10.0)
        
        post_selectors = [
            '[data-view-name="feed-full-update"]',
            '[role="listitem"]',
            '.reusable-search__result-container',
            '.search-results-container [role="listitem"]',
            '.feed-shared-update-v2', 
            '.occludable-update'
        ]
        content_selectors = [
            '[data-view-name="feed-commentary"]',
            '[data-testid="expandable-text-box"]',
            '.feed-shared-update-v2__description', 
            '.update-components-text-view'
        ]

        max_scrolls = (limit // 3) + 3
        consecutive_empty_scrolls = 0
        
        for i in range(max_scrolls):
            count_before = len(results)
            for post_selector in post_selectors:
                js_code = r"""(selector) => {
                    const posts = [];
                    const items = document.querySelectorAll(selector);
                    
                    // Sequential scroll for each item to ensure hydration
                    // We only scroll the most recent items to trigger lazy loading
                    const startIndex = Math.max(0, items.length - 12);
                    for (let i = startIndex; i < items.length; i++) {
                        try {
                            items[i].scrollIntoView({block: 'center'});
                        } catch(e) {}
                    }
                    
                    for (const item of items) {
                        try {
                            const seeMore = item.querySelector('.feed-shared-inline-show-more-text__button');
                            if (seeMore) seeMore.click();
                        } catch(e) {}

                        // Try standard content selectors
                        let content = "";
                        const contentSels = [
                            '[data-view-name="feed-commentary"]',
                            '[data-testid="expandable-text-box"]', 
                            '.feed-shared-update-v2__description', 
                            '.update-components-text-view',
                            '.feed-shared-text',
                            '.feed-shared-update-v2__commentary'
                        ];
                        for (const sel of contentSels) {
                            const el = item.querySelector(sel);
                            if (el && el.innerText.trim().length > 10) {
                                content = el.innerText.trim();
                                break;
                            }
                        }
                        
                        if (!content || content.length < 15) {
                            continue;
                        }
                        
                        let sanitizedContent = content
                            .replace(/[\r\n]+/g, ' ')
                            .replace(/"/g, "'")
                            .replace(String.fromCharCode(92), '/') 
                            .trim();
                        
                        if (sanitizedContent.length > 1500) {
                            sanitizedContent = sanitizedContent.substring(0, 1500) + "... [TRUNCATED FOR AGENT]";
                        }
                        
                        let name = "Unknown";
                        let profile = "";
                        let title = "LinkedIn User";
                        
                        const links = Array.from(item.querySelectorAll('a[href*="/in/"]'));
                        for (const a of links) {
                            if (a.getAttribute('data-view-name') === 'feed-actor-image') continue;
                            const textLines = a.innerText.split('\n').map(l => l.trim()).filter(l => l);
                            if (textLines.length > 0) {
                                name = textLines[0].split('•')[0].trim();
                                if (name === "LinkedIn Member") {
                                    // Try another link or fallback
                                    continue;
                                }
                                profile = a.href;
                                if (textLines.length > 1) title = textLines[1].split('•')[0].trim();
                                break;
                            }
                        }
                        
                        // Fallback name if still unknown or "LinkedIn Member"
                        if (name === "Unknown" || name === "LinkedIn Member" || name.length < 2) {
                            const nameEl = item.querySelector('[data-view-name="feed-actor-name"], .update-components-actor__name, .hoverable-link-text');
                            if (nameEl) {
                                let foundName = (nameEl.getAttribute('title') || nameEl.innerText).split('•')[0].trim();
                                if (foundName && foundName.length > 1) name = foundName;
                            }
                        }
                        
                        if (title === "LinkedIn User") {
                            const titleEl = item.querySelector('.update-components-actor__description, .update-components-actor__headline, .t-14.t-black--light.t-normal');
                            if (titleEl) title = titleEl.innerText.trim();
                        }
                        
                        let postLink = "";
                        const pLinkEl = item.querySelector('a[href*="/feed/update/urn:li:"]');
                        if (pLinkEl) postLink = pLinkEl.href;
                        
                        posts.push({
                            content: sanitizedContent,
                            author: { 
                                name: name.replace(/"/g, "'").replace(String.fromCharCode(92), '/'), 
                                title: title.replace(/[\r\n]+/g, ' ').replace(/"/g, "'").replace(String.fromCharCode(92), '/').trim(), 
                                profile_link: profile 
                            },
                            post_link: postLink
                        });
                    }
                    return posts;
                }"""

                raw_posts = await bm.page.evaluate(js_code, post_selector)

                if not raw_posts: continue
                
                tool_log(f"Extracted {len(raw_posts)} items via selector '{post_selector}'")
                for post_data in raw_posts:
                    author_name = post_data["author"]["name"]
                    content_text = post_data["content"]
                    
                    unique_key = f"{author_name}:{content_text[:100]}"
                    if unique_key not in all_content:
                        # Log more clearly for the user ONLY for new posts
                        content_preview = content_text[:60].replace('\n', ' ') + "..."
                        print(f"   [LinkedIn] Found: {author_name} - {content_preview}")
                        
                        all_content.add(unique_key)
                        results.append(post_data)
                        if fw:
                            fw.event_bus.emit("scraper:post_discovered", {"post": post_data})
                            fw.event_bus.emit("tool:log", {"message": f"Extracted: {author_name}"})
                
                # If we found posts with one selector, don't try others in this scroll
                break

            new_posts = len(results) - count_before
            if new_posts == 0:
                consecutive_empty_scrolls += 1
            else:
                consecutive_empty_scrolls = 0

            if len(results) >= limit: break
            
            if consecutive_empty_scrolls >= 2 and len(results) == 0 and i > 0:
                # Debug Check: If we found nothing after scrolling twice, capture HTML
                tool_log("WARNING: Found 0 results after 2 scrolls. Capturing debug HTML...")
                debug_path = f"debug_search_fail_{int(asyncio.get_event_loop().time())}.html"
                try:
                    html = await bm.page.content()
                    with open(debug_path, "w") as f: f.write(html)
                    tool_log(f"Debug HTML saved to: {debug_path}")
                except: pass

            if consecutive_empty_scrolls >= 3:
                tool_log(f"Stopping search: 3 consecutive scrolls found no new posts.")
                break
                
            tool_log(f"Scroll {i+1}: Found {new_posts} new posts. Total: {len(results)}")
            
            # More persistent scroll jump
            await bm.page.evaluate('''() => {
               window.scrollBy(0, 1500);
               const main = document.querySelector('.scaffold-layout__main') || document.querySelector('main');
               if (main) main.scrollBy(0, 1500);
               const lazyColumn = document.querySelector('[data-testid="lazy-column"]');
               if (lazyColumn) lazyColumn.scrollBy(0, 1500);
            }''')
            
            # Use PageDown multiple times
            for _ in range(3):
                await bm.page.keyboard.press("PageDown")
                await asyncio.sleep(0.5)
            
            await asyncio.sleep(random.uniform(5.0, 8.0))
            
            if new_posts == 0:
                # If still no results, maybe it's the very bottom or a slow load
                await bm.page.keyboard.press("End")
                await asyncio.sleep(3)
            
    tool_log(f"Completed search. Found {len(results)} posts.")
    return results[:limit]

async def get_linkedin_profile_details(profile_url: str = None, session_path: str = "linkedin_session.json", **kwargs) -> Dict:
    """Extract detailed information from a LinkedIn profile. Requires 'profile_url'."""
    fw = kwargs.get('framework')
    def tool_log(msg):
        if fw: fw.event_bus.emit("tool:log", {"agent": "LinkedInTool", "message": msg})
        else: print(f"   [Tool] {msg}")

    # Resilience: If agent uses 'query' or other names instead of 'profile_url'
    if not profile_url:
        profile_url = kwargs.get('query') or kwargs.get('url') or kwargs.get('profile') or kwargs.get('link')
        
    if not profile_url or not str(profile_url).startswith('http'):
        return {"error": "INVALID_URL", "message": f"'{profile_url}' is not a valid LinkedIn URL. Please provide a full URL."}

    tool_log(f"Visiting profile: {profile_url}")
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        active_session = _resolve_session_path(session_path)
        if active_session: await bm.load_session(active_session)
        else: return {"error": "SESSION_MISSING", "message": "Session file not found."}

        await bm.page.goto(profile_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        details = {"name": "Unknown", "headline": "Unknown", "about": "", "experience": []}
        try:
            name_el = bm.page.locator('h1.text-heading-xlarge').first
            if await name_el.count() > 0: details["name"] = (await name_el.inner_text()).strip()
            headline_el = bm.page.locator('.text-body-medium.break-words').first
            if await headline_el.count() > 0: details["headline"] = (await headline_el.inner_text()).strip()
            about_section = bm.page.locator('#about').locator('..').locator('.display-flex.ph5.pv3').first
            if await about_section.count() > 0: details["about"] = (await about_section.inner_text()).strip()
            exp_items = await bm.page.locator('.experience-group-positions, .pvs-list__paged-list-item').all()
            for item in exp_items[:3]:
                text = await item.inner_text()
                if text.strip(): details["experience"].append(text.strip().replace('\n', ' | '))
        except Exception as e: 
            tool_log(f"Error scraping profile: {e}")
            details["error"] = str(e)
        return details

async def get_linkedin_company_details(company_url: str = None, session_path: str = "linkedin_session.json", **kwargs) -> Dict:
    """Extract detailed information from a LinkedIn Company page. Requires 'company_url'."""
    fw = kwargs.get('framework')
    def tool_log(msg):
        if fw: fw.event_bus.emit("tool:log", {"agent": "LinkedInTool", "message": msg})
        else: print(f"   [Tool] {msg}")

    # Resilience: If agent uses 'query' or other names instead of 'company_url'
    if not company_url:
        company_url = kwargs.get('query') or kwargs.get('url') or kwargs.get('company') or kwargs.get('link')

    if not company_url or not str(company_url).startswith('http'):
        return {"error": "INVALID_URL", "message": f"'{company_url}' is not a valid LinkedIn URL. Please provide a full URL."}

    if "/about" not in company_url: company_url = company_url.rstrip('/') + "/about/"
    tool_log(f"Visiting company: {company_url}")
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        active_session = _resolve_session_path(session_path)
        if active_session: await bm.load_session(active_session)
        else: return {"error": "SESSION_MISSING", "message": "Session file not found."}

        await bm.page.goto(company_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        details = {"name": "Unknown", "industry": "Unknown", "size": "Unknown", "description": ""}
        try:
            name_el = bm.page.locator('h1.org-top-card-summary__title').first
            if await name_el.count() > 0: details["name"] = (await name_el.inner_text()).strip()
            desc_el = bm.page.locator('.org-about-us-organization-description__text').first
            if await desc_el.count() > 0: details["description"] = (await desc_el.inner_text()).strip()
            grid_items = await bm.page.locator('.org-page-details__definition-term').all()
            for i, item in enumerate(grid_items):
                term = (await item.inner_text()).lower()
                value_el = bm.page.locator('.org-page-details__definition-text').nth(i)
                if "industry" in term: details["industry"] = (await value_el.inner_text()).strip()
                elif "company size" in term: details["size"] = (await value_el.inner_text()).strip()
        except Exception as e: 
            tool_log(f"Error scraping company: {e}")
            details["error"] = str(e)
        return details
