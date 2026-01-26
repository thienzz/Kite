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

async def create_linkedin_session(session_path: str = "linkedin_session.json"):
    """Create a LinkedIn session file with bulletproof detection."""
    print("="*60)
    print("LinkedIn Session Creator (Integrated)")
    print("="*60)
    
    # Use headless=False so the user can see and interact
    async with BrowserManager(headless=False) as browser:
        print("Opening LinkedIn...")
        await browser.page.goto("https://www.linkedin.com/login")
        
        print("\n🔐 Please log in to LinkedIn manually in the opened browser.")
        print("   I will auto-detect once you are on the Feed or any main page.")
        print("\n⏳ Monitoring login status (timeout 5 mins)...\n")
        
        start_time = asyncio.get_event_loop().time()
        logged_in = False
        
        while (asyncio.get_event_loop().time() - start_time) < 300: # 5 minutes
            if await is_really_logged_in(browser.page):
                logged_in = True
                break
            await asyncio.sleep(2)
            
        if logged_in:
            print(f"\n✅ SUCCESS: Login detected!")
            print(f"💾 Saving session to {session_path}...")
            await asyncio.sleep(3)
            await browser.save_session(session_path)
            print("\n✅ DONE!")
        else:
            print("\n❌ FAILED: Login not detected within 5 minutes.")
            return False
    return True

def check_red_flags(text: str, red_flags: List[str]) -> Dict:
    """
    Check for exclusion signals (Red Flags) in the post text.
    Configurable via the 'red_flags' parameter.
    """
    text_lower = text.lower()
    found = [flag for flag in red_flags if flag.lower() in text_lower]
    
    return {
        "is_job_post": len(found) > 0,
        "flags_found": found
    }

def evaluate_profile_sync(profile: Dict, config: Dict) -> Dict:
    """
    Evaluate the post author's profile based on provided configuration.
    Config expected to have: 'excluded_titles', 'qualified_titles', 'excluded_locations', 'excluded_name_patterns'.
    """
    title = profile.get("title", "").lower()
    location = profile.get("location", "").lower()
    name = profile.get("name", "").lower()

    excluded_titles = config.get("excluded_titles", [])
    qualified_titles = config.get("qualified_titles", [])
    excluded_locations = config.get("excluded_locations", [])
    excluded_name_patterns = config.get("excluded_name_patterns", [])
    
    is_qualified_title = any(qt.lower() in title for qt in qualified_titles) and \
                         not any(et.lower() in title for et in excluded_titles)
    
    is_qualified_location = not any(loc.lower() in location for loc in excluded_locations)
    
    is_excluded_name = any(pattern.lower() in name for pattern in excluded_name_patterns)

    return {
        "is_qualified": is_qualified_title and is_qualified_location and not is_excluded_name,
        "reason": f"Title: {title}, Location: {location}"
    }

def detect_buying_intent(text: str, gold_keywords: List[str]) -> Dict:
    """
    Identify 'Gold' keywords indicating buying intent.
    Configurable via the 'gold_keywords' parameter.
    """
    text_lower = text.lower()
    found = [kw for kw in gold_keywords if kw.lower() in text_lower]
    
    return {
        "has_intent": len(found) > 0,
        "intent_keywords": found
    }

async def search_linkedin_posts(query: str, limit: int = 30, session_path: str = "linkedin_session.json", **kwargs) -> List[Dict]:
    """
    Search for LinkedIn posts.
    - query: The search string (Boolean supported).
    - limit: Max results to return.
    """
    # LLMs frequently try to pass 'keywords' as a separate list. Merge it.
    if "keywords" in kwargs:
        if isinstance(kwargs["keywords"], list):
            query += " " + " ".join(kwargs["keywords"])
        elif isinstance(kwargs["keywords"], str):
            query += " " + kwargs["keywords"]
    
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    results = []
    all_content = set()
    
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        if os.path.exists(session_path):
            await bm.load_session(session_path)
        else:
            print(f"   [Tool] Error: {session_path} not found.")
            return []

        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.linkedin.com/search/results/content/?keywords={encoded_query}&origin=GLOBAL_SEARCH_HEADER"
        
        await bm.page.goto(search_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # Try to click the 'Posts' filter if we are on a mixed results page
        try:
            posts_filter = bm.page.locator('button:has-text("Posts")').first
            if await posts_filter.is_visible():
                await posts_filter.click()
                await asyncio.sleep(3)
        except: pass
        
        # Search for post containers first to keep author and content together
        post_selectors = [
            '.feed-shared-update-v2',
            '.occludable-update',
            'div[data-urn]'
        ]
        
        content_selectors = [
            '.feed-shared-update-v2__description',
            '.update-components-text-view',
            '[data-testid="expandable-text-box"]'
        ]

        max_scrolls = (limit // 3) + 3
        for i in range(max_scrolls):
            # 1. Strategy A: Container-based (High Quality)
            for post_selector in post_selectors:
                posts = await bm.page.locator(post_selector).all()
                for post_el in posts:
                    try:
                        content_text = ""
                        for c_sel in content_selectors:
                            c_el = post_el.locator(c_sel).first
                            if await c_el.count() > 0:
                                content_text = await c_el.inner_text()
                                break
                        
                        if not content_text or len(content_text.strip()) < 30:
                            continue
                            
                        author_name = "Unknown"
                        author_title = "LinkedIn User"
                        profile_link = ""
                        post_link = ""
                        
                        # 2. Get Author and Profile Link
                        name_container = post_el.locator('.update-components-actor__container, .app-aware-link').first
                        if await name_container.count() > 0:
                            author_name = (await name_container.locator('.update-components-actor__name, span > span').first.inner_text()).strip()
                            # Profile link is usually the href of the name container or a child link
                            profile_link_el = post_el.locator('a.app-aware-link[href*="/in/"]').first
                            if await profile_link_el.count() > 0:
                                profile_link = await profile_link_el.get_attribute("href")
                        
                        title_el = post_el.locator('.update-components-actor__description, .update-components-actor__headline').first
                        if await title_el.count() > 0:
                            author_title = (await title_el.inner_text()).strip()

                        # 3. Get Post Link
                        # Post link often found in the relative time link or via data-urn
                        time_link = post_el.locator('a[href*="/feed/update/urn:li:activity:"]').first
                        if await time_link.count() > 0:
                            post_link = await time_link.get_attribute("href")
                        
                        if post_link and not post_link.startswith("http"):
                            post_link = "https://www.linkedin.com" + post_link

                        unique_key = f"{author_name}:{content_text[:100]}"
                        if unique_key not in all_content:
                            all_content.add(unique_key)
                            results.append({
                                "content": content_text.strip(),
                                "author": {
                                    "name": author_name, 
                                    "title": author_title,
                                    "profile_link": profile_link
                                },
                                "post_link": post_link
                            })
                    except: continue

            # 2. Strategy B: Broad (Fallback) - Only if we are low on results
            if len(results) < limit:
                broad_selectors = ['[data-testid="expandable-text-box"]', '.update-components-text-view']
                for b_sel in broad_selectors:
                    elements = await bm.page.locator(b_sel).all()
                    for el in elements:
                        try:
                            text = (await el.inner_text()).strip()
                            if len(text) > 60 and text[:100] not in str(all_content):
                                results.append({
                                    "content": text,
                                    "author": {"name": "LinkedIn User", "title": "Potential Lead (Broad Match)"}
                                })
                                all_content.add(text[:100])
                        except: continue

            if len(results) >= limit:
                break
            
            # Smart scrolling
            await bm.page.mouse.wheel(0, random.randint(800, 1300))
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
    if not results:
        print("   [Tool] WARNING: No posts found. LinkedIn might be blocking or selectors changed.")
    
    print(f"   [Tool] Found {len(results)} posts.")
    return results[:limit]

async def get_linkedin_profile_details(profile_url: str, session_path: str = "linkedin_session.json") -> Dict:
    """
    Extract detailed information from a LinkedIn profile.
    """
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        if os.path.exists(session_path):
            await bm.load_session(session_path)
        else:
            return {"error": "Session not found"}

        await bm.page.goto(profile_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        details = {
            "name": "Unknown",
            "headline": "Unknown",
            "about": "",
            "experience": []
        }

        try:
            # 1. Name & Headline
            name_el = bm.page.locator('h1.text-heading-xlarge').first
            if await name_el.count() > 0:
                details["name"] = (await name_el.inner_text()).strip()
            
            headline_el = bm.page.locator('.text-body-medium.break-words').first
            if await headline_el.count() > 0:
                details["headline"] = (await headline_el.inner_text()).strip()

            # 2. About
            about_section = bm.page.locator('#about').locator('..').locator('.display-flex.ph5.pv3').first
            if await about_section.count() > 0:
                details["about"] = (await about_section.inner_text()).strip()

            # 3. Experience (Quick look at top 3)
            exp_items = await bm.page.locator('.experience-group-positions, .pvs-list__paged-list-item').all()
            for item in exp_items[:3]:
                text = await item.inner_text()
                if text.strip():
                    details["experience"].append(text.strip().replace('\n', ' | '))
                    
        except Exception as e:
            details["error"] = str(e)

        return details

async def get_linkedin_company_details(company_url: str, session_path: str = "linkedin_session.json") -> Dict:
    """
    Extract detailed information from a LinkedIn Company page.
    """
    if "/about" not in company_url:
        company_url = company_url.rstrip('/') + "/about/"

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    async with BrowserManager(headless=True, user_agent=ua) as bm:
        if os.path.exists(session_path):
            await bm.load_session(session_path)
        else:
            return {"error": "Session not found"}

        await bm.page.goto(company_url, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        details = {
            "name": "Unknown",
            "industry": "Unknown",
            "size": "Unknown",
            "description": ""
        }

        try:
            name_el = bm.page.locator('h1.org-top-card-summary__title').first
            if await name_el.count() > 0:
                details["name"] = (await name_el.inner_text()).strip()

            # About/Description
            desc_el = bm.page.locator('.org-about-us-organization-description__text').first
            if await desc_el.count() > 0:
                details["description"] = (await desc_el.inner_text()).strip()

            # Grid details (Industry, Size, etc)
            grid_items = await bm.page.locator('.org-page-details__definition-term').all()
            for i, item in enumerate(grid_items):
                term = (await item.inner_text()).lower()
                value_el = bm.page.locator('.org-page-details__definition-text').nth(i)
                if "industry" in term:
                    details["industry"] = (await value_el.inner_text()).strip()
                elif "company size" in term:
                    details["size"] = (await value_el.inner_text()).strip()
                    
        except Exception as e:
            details["error"] = str(e)

        return details
