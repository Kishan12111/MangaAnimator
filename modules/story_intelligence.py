"""
Story Intelligence Engine Module

AI-powered story understanding and narration generation.
Uses LLMs to analyze manga text and produce narration scripts.
Supports Google Gemini (free tier), OpenAI GPT, and placeholder modes.
"""

import base64
import json
import logging
import os
import re
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from interfaces.base_story_intelligence import (
    BaseStoryIntelligence,
    StoryIntelligenceInput,
    StoryIntelligenceOutput,
    NarrativeTone
)

logger = logging.getLogger(__name__)


class StoryIntelligenceEngine(BaseStoryIntelligence):
    """
    Concrete implementation of story intelligence.
    
    Supports multiple LLM backends:
    - Google Gemini (free tier, recommended)
    - OpenAI GPT models
    - Placeholder mode for testing
    """
    
    WORDS_PER_SECOND = 2.5  # Average speaking rate
    
    def __init__(self, model_name: str = "placeholder", api_key: Optional[str] = None, anime_title: str = ""):
        self._model_name = model_name
        self._model_params: Dict[str, Any] = {}
        self._client = None
        self._api_key = api_key
        self._anime_title = anime_title
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model."""
        if self._model_name == "placeholder":
            logger.info("Using placeholder story intelligence (no LLM)")
            return
        
        if self._model_name.startswith("gemini"):
            try:
                from google import genai as genai_client
                
                api_key = self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning("No Gemini API key found. Set GEMINI_API_KEY env var or pass api_key. Falling back to placeholder.")
                    self._model_name = "placeholder"
                    return
                
                # Use new google.genai SDK
                model_id = self._model_name if "/" not in self._model_name else self._model_name
                if model_id == "gemini":
                    model_id = "gemini-2.5-flash"
                
                self._client = genai_client.Client(api_key=api_key)
                self._model_name = model_id
                logger.info(f"Initialized Google Gemini with model: {model_id}")
                
            except ImportError:
                logger.warning("google-genai package not installed. Run: pip install google-genai")
                self._model_name = "placeholder"
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self._model_name = "placeholder"
        
        elif self._model_name.startswith("gpt"):
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
                logger.info(f"Initialized OpenAI client with model: {self._model_name}")
            except ImportError:
                logger.warning("openai package not installed. Falling back to placeholder.")
                self._model_name = "placeholder"
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self._model_name = "placeholder"
        else:
            logger.warning(f"Unknown model: {self._model_name}. Using placeholder.")
            self._model_name = "placeholder"
    
    def set_model(self, model_name: str, **model_params) -> None:
        """Set the underlying LLM model."""
        self._model_name = model_name
        self._model_params = model_params
        self._initialize_model()
    
    def analyze(self, input_data: StoryIntelligenceInput, panel_images: Optional[List[np.ndarray]] = None) -> StoryIntelligenceOutput:
        """Analyze the story from extracted panel texts and optionally images."""
        logger.info("Analyzing story from panel texts")
        
        if self._model_name == "placeholder":
            return self._placeholder_analyze(input_data)
        
        try:
            # If we have panel images and Gemini, use vision-based analysis
            if panel_images and self._model_name.startswith("gemini"):
                return self._vision_analyze(input_data, panel_images)
            return self._llm_analyze(input_data)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}. Falling back to placeholder.")
            return self._placeholder_analyze(input_data)
    
    def _vision_analyze(self, input_data: StoryIntelligenceInput, panel_images: List[np.ndarray]) -> StoryIntelligenceOutput:
        """Perform Gemini Vision-based story analysis using panel images."""
        import PIL.Image
        
        max_words = int(input_data.max_duration_seconds * self.WORDS_PER_SECOND)
        
        # Select up to 16 representative panel images for Gemini
        if len(panel_images) > 16:
            indices = np.linspace(0, len(panel_images) - 1, 16, dtype=int)
            selected_images = [panel_images[i] for i in indices]
            selected_indices = [input_data.panel_indices[i] for i in indices]
        else:
            selected_images = panel_images
            selected_indices = list(input_data.panel_indices)
        
        # Convert numpy images to PIL for Gemini
        pil_images = []
        for img in selected_images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            # Resize large images to save tokens (but keep them big enough
            # for Gemini to read speech bubble text clearly)
            h, w = img.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            pil_images.append(PIL.Image.fromarray(img))
        
        # Also include OCR text if available
        panel_text_list = "\n".join([
            f"Panel {idx}: {text}" 
            for idx, text in zip(input_data.panel_indices, input_data.panel_texts)
            if text.strip()
        ])
        
        # Build anime-specific context
        chapter_title = input_data.metadata.get('chapter_title', '') if input_data.metadata else ''
        anime_context = ""
        if self._anime_title or chapter_title:
            # The anime_title may contain batch context (previous chapter recap,
            # cliffhanger instructions, stage-setting cues) injected by BatchPipeline.
            # Pass it through verbatim so the LLM honours those instructions.
            anime_context = f"""ANIME/MANGA CONTEXT:
Series: {self._anime_title or 'Unknown Manga'}
{f'Chapter/File: {chapter_title}' if chapter_title else ''}

You are an expert on this anime/manga. You know every character, their powers,
their relationships, and every story arc. Use this knowledge for accurate narration.
If batch instructions appear above (PREVIOUS CHAPTER CONTEXT, CLIFFHANGER ENDING,
STAGE-SETTING), follow them precisely.

"""
        
        prompt = f"""ROLE:
You are a world-class anime/manga recap narrator. You adapt your voice to whatever
the story demands — intense when the stakes are high, tender when a character
breaks down, witty when the scene is funny, and urgent when there's a fight.
You never force one energy on every scene. You READ THE ROOM.
{anime_context}
OBJECTIVE:
Turn these manga panels into narration that feels like the perfect voice-over for
this specific moment in the story. Match the emotional temperature exactly.

CONTEXT:
This is a short-form vertical video (YouTube Shorts / TikTok / Reels).
This chapter is part of an ongoing series — write as a continuation, not an intro.
The opening line must hook the viewer within 1 second.
VIDEO LENGTH: This video will be approximately {int(input_data.max_duration_seconds)} seconds.
Adjust your narration length accordingly — more panels = longer narration.

STAGE-SETTING PANELS:
If the story context calls for a stage-setting moment (e.g. introducing a powerful
character, establishing a new location after a dramatic shift), pick the most
visually striking panel and write an opening line that poses a question or sets
the mood over that image (e.g. "Why is Saitama considered the strongest?" over
a wide shot of Saitama). ONLY do this when the visuals genuinely support it —
maybe 1 in every 3-4 episodes. Otherwise jump straight into the action.

INPUT:
You are looking at {len(pil_images)} manga panels in reading order.

CRITICAL — READ THE IMAGES:
1. **READ every speech bubble, thought bubble, sound effect, and caption** in each panel.
   This is manga — the text IS the story. You MUST read and understand all dialogue.
2. Identify who is speaking in each panel based on the bubble tails.
3. Understand the ACTION happening in each panel from the art.
4. Track the EMOTIONAL SHIFTS across panels (calm → tense → explosive → aftermath etc.).

{f"Supplementary OCR text (may be incomplete/noisy — trust your own reading of the images):{chr(10)}{panel_text_list}" if panel_text_list else "(No supplementary OCR — read all text directly from the panel images above.)"}
Panel indices available: {selected_indices}

STEP 1 — UNDERSTAND THE STORY (think through this carefully):
Read every panel. For each one, note:
  a) What dialogue/text appears in speech bubbles?
  b) What action is happening?
  c) Which characters are present and what are their expressions/body language?
  d) How does this panel connect to the previous one?
Then identify the STORY ARC across all panels:
  - What is the situation at the START?
  - What CONFLICT or EVENT unfolds?
  - How does it RESOLVE or what CLIFFHANGER does it end on?
  - What is the dominant emotional tone?

Tone guide:
- Battle / Action → sharp, urgent, breathless pacing
- Death / Loss / Sacrifice → measured, heavy, let weight land on its own
- Romance / Bond → warm, genuine, understated — never corny
- Comedy / Absurd → dry, deadpan, let the joke breathe
- Mystery / Suspense → slow build, hints, unanswered questions
- Triumph / Victory → rising energy, earn the payoff
- Horror / Dread → quiet tension, unsettling calm
Most scenes blend 2-3 of these. Find the mix.

STEP 2 — WRITE THE NARRATION:

You must narrate the SPECIFIC EVENTS from these panels. Reference what actually
happens — character names, dialogue context, actions, reactions, and plot points.
DO NOT write generic narration that could apply to any manga. The viewer is looking
at THESE panels — your narration must describe what THEY show.

RULES:
* WORD COUNT: {max_words} words (±15). Scale narration to cover ALL panels shown.
  For 10+ panels, write a FULL narration that covers each key moment — don't skip content.
* HOOK FIRST — open with a compelling line that fits the TONE.
  Action hook: "The ground splits open." | Emotional hook: "She never got to say goodbye."
  Mystery hook: "Something was wrong with that smile." | Comedy hook: "This man just cooked ramen during a war."
* Use SHORT sentences. 5-15 words. Vary rhythm based on tone.
* Sound NATURAL — like a real person telling a story to a friend, not performing.
* Let QUIET moments be quiet. Not everything needs exclamation energy.
* Build tension through PACING, not adjectives.
* NEVER list events mechanically ("First X, then Y, then Z").
* NEVER use narrator clichés: "little did he know", "but that's not all",
  "things are about to get crazy", "buckle up", "brace yourself".
* NEVER address the viewer ("you won't believe", "get ready").
* Use CHARACTER NAMES once, then switch to pronouns.
* Write for the EAR. Every line must sound natural when spoken aloud.
* End on a moment that makes people want the next episode — but match the tone.
  A quiet ending can be just as powerful as a dramatic one.

WHAT MAKES NARRATION BAD (AVOID):
- Same hyped energy for a funeral and a fight scene
- Sounding like a Wikipedia article or book report
- Over-explaining what the viewer can already see
- Fancy vocabulary when simple words hit harder
- Forcing excitement where the story calls for stillness
- **GENERIC narration that could apply to ANY manga** — this is the #1 failure mode
- Saying things like "the story unfolds" or "things take a dramatic turn" without
  naming WHO does WHAT. ALWAYS be specific.

WHAT MAKES NARRATION ADDICTIVE (DO THIS):
- Reference SPECIFIC events: "Genos fires the incineration cannon. Point blank."
  NOT: "The hero unleashes a powerful attack."
- Use dramatic "…" pauses before reveals: "And standing behind him… was Garou."
- Drop character names at impactful moments for weight.
- Use sentence fragments for action beats: "One punch. That's all it took."
- Let emotional lines breathe — one short sentence, then silence.
- Build MOMENTUM — start controlled, escalate through the middle, peak near the end.
- Plant micro-hooks throughout: "But that wasn't even the worst part."
- End on an UNRESOLVED beat — a question unanswered, a threat looming, an emotion unspoken.
  The viewer should feel INCOMPLETE without the next episode.
- Match the RHYTHM of the action: rapid short sentences for fights, longer flowing
  sentences for emotional moments, staccato fragments for shock/reveals.

PANEL SELECTION:
Select the panels that best represent the narration's key moments.
For short chapters (≤8 panels): select most of them.
For longer chapters (9-16 panels): select at least 6-10 key moments.
Cover the FULL story arc — beginning, middle, climax, and end.

FAILSAFE:
If OCR text is noisy or partially readable, infer meaning from visual context
and the manga's known lore. Never fall back to generic filler.
OUTPUT FORMAT (STRICT JSON, no markdown, no explanation):
{{
    "narration_script": "the narration text here...",
    "intro_hook": "A short provocative sentence (8-20 words) designed to STOP someone from scrolling. Must reference a SPECIFIC character or event from THIS chapter. Frame it as a question, a bold claim, or an unresolved tension. Examples: 'Garou stands before the entire S-Class. Alone. And he's smiling.' | 'She gave up everything. And it still wasn't enough.' | 'They told him he was the weakest. They were wrong.' BAD examples (too generic): 'The battle begins.' | 'Everything is about to change.' | 'Chapter 120.' The hook must sound like something a real person would say to get your attention.",
    "intro_panel_index": "(integer) the index of the single most visually STRIKING panel — a dramatic close-up, a powerful pose, a beautiful character shot, or a jaw-dropping action frame. This panel will be shown as the intro background image. Pick the one that would make someone STOP scrolling. Prefer panels showing a character's face, a dramatic pose, or an emotional moment.",
    "key_events": ["brief description of each major plot beat — be specific, use names and actions"],
    "selected_panel_indices": [panel indices from {selected_indices}],
    "tone": "one of: dramatic | hype | emotional | intense | somber | comedic | suspenseful | triumphant",
    "characters": [{{"name": "character name", "gender": "male | female | unknown", "features": "hair color/style, outfit, distinctive visual features", "role": "protagonist | antagonist | supporting"}}]
}}

IMPORTANT: ONLY output valid JSON. No explanations, no markdown fences.
REMINDER: Your narration MUST reference the SPECIFIC events visible in these panels.
Generic narration = failure."""

        # Call Gemini with images + prompt via new google.genai SDK
        from google.genai import types
        
        content_parts = list(pil_images) + [prompt]
        
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=8000,
            )
        )
        
        response_text = response.text
        logger.debug(f"Vision analysis response: {response_text[:500]}")
        
        # Clean markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text.strip())
        
        result = self._parse_json_response(response_text)
        
        # Support both key names: narration_script (new) and summary_script (legacy)
        script = result.get("narration_script", "") or result.get("summary_script", "")
        selected_panels = result.get("selected_panel_indices", []) or result.get("selected_panels", selected_indices)
        tone = result.get("tone", "neutral")
        key_events = result.get("key_events", [])
        characters_raw = result.get("characters", [])
        intro_hook = result.get("intro_hook", "")
        intro_panel_index = result.get("intro_panel_index", None)
        
        # Normalize characters to string list
        characters = []
        character_details = []
        for c in characters_raw:
            if isinstance(c, dict):
                characters.append(c.get("name", "Unknown"))
                character_details.append(c)
            else:
                characters.append(str(c))
        
        # Try to convert intro_panel_index to int
        if intro_panel_index is not None:
            try:
                intro_panel_index = int(intro_panel_index)
            except (ValueError, TypeError):
                intro_panel_index = None
        
        script = self.adjust_for_duration(script, input_data.max_duration_seconds)
        word_count = len(script.split())
        estimated_duration = word_count / self.WORDS_PER_SECOND
        
        return StoryIntelligenceOutput(
            summary_script=script,
            selected_panels=selected_panels,
            tone=tone,
            key_events=key_events,
            characters=characters,
            estimated_duration=estimated_duration,
            confidence=0.9,
            intro_hook=intro_hook,
            metadata={
                'model': self._model_name,
                'analysis_type': 'vision',
                'character_details': character_details,
                'intro_panel_index': intro_panel_index,
            }
        )
    
    def _llm_analyze(self, input_data: StoryIntelligenceInput) -> StoryIntelligenceOutput:
        """Perform LLM-based story analysis."""
        # Prepare the prompt
        panel_text_list = "\n".join([
            f"Panel {idx}: {text}" 
            for idx, text in zip(input_data.panel_indices, input_data.panel_texts)
            if text.strip()
        ])
        
        max_words = int(input_data.max_duration_seconds * self.WORDS_PER_SECOND)
        
        anime_context = ""
        if self._anime_title:
            anime_context = f"""ANIME/MANGA CONTEXT:
{self._anime_title}
You are an expert on this anime/manga. Use character names, powers, and lore correctly.
If batch instructions appear above (PREVIOUS CHAPTER CONTEXT, CLIFFHANGER ENDING,
STAGE-SETTING), follow them precisely.

"""
        
        prompt = f"""ROLE:
You are a world-class anime/manga recap narrator. You adapt your tone to whatever
the story demands — intense for fights, tender for loss, witty for comedy, urgent
for danger. You READ THE ROOM and never force one energy on every scene.
{anime_context}
PANEL TEXTS (in reading order):
{panel_text_list if panel_text_list else "(No text detected)"}

STEP 1 — Read the text. What is the dominant feeling?
- Battle → sharp, urgent | Death/Loss → measured, heavy | Comedy → dry, deadpan
- Romance → warm, understated | Mystery → slow build | Triumph → rising energy
Find the mix and write to it.

STEP 2 — Write narration:
* WORD COUNT: {max_words} words (±15). Cover the full story arc — don't skip content.
* HOOK FIRST — a compelling opening line that fits the tone.
* SHORT sentences. 5-15 words. Vary rhythm.
* Sound NATURAL, like telling a friend. Never forced hype.
* Let quiet moments be quiet. Match the emotional temperature.
* NEVER list events mechanically. NEVER use clichés.
* Write for the EAR — must sound natural spoken aloud.
* End on a moment that earns the next episode.

OUTPUT FORMAT (STRICT JSON):
{{
    "narration_script": "the narration...",
    "intro_hook": "A single atmospheric sentence (8-15 words) that sets the mood. NOT a summary — a mood-setter for the title card.",
    "key_events": ["event1", "event2"],
    "selected_panel_indices": [panel indices],
    "tone": "dramatic | hype | emotional | intense | somber | comedic | suspenseful | triumphant",
    "characters": ["character names"]
}}

ONLY output valid JSON, no explanations."""

        response = self._call_llm(prompt)
        
        try:
            # Clean markdown code blocks
            clean = re.sub(r'```json\s*', '', response)
            clean = re.sub(r'```\s*$', '', clean.strip())
            
            # Parse JSON response (with truncation repair)
            result = self._parse_json_response(clean)
            
            script = result.get("narration_script", "") or result.get("summary_script", "")
            selected_panels = result.get("selected_panel_indices", []) or result.get("selected_panels", [])
            tone = result.get("tone", "neutral")
            key_events = result.get("key_events", [])
            characters = result.get("characters", [])
            intro_hook = result.get("intro_hook", "")
            
            # Adjust script for duration
            script = self.adjust_for_duration(
                script, 
                input_data.max_duration_seconds
            )
            
            # Estimate duration
            word_count = len(script.split())
            estimated_duration = word_count / self.WORDS_PER_SECOND
            
            return StoryIntelligenceOutput(
                summary_script=script,
                selected_panels=selected_panels,
                tone=tone,
                key_events=key_events,
                characters=characters,
                estimated_duration=estimated_duration,
                confidence=0.8,
                intro_hook=intro_hook,
                metadata={'model': self._model_name}
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise
    
    def _parse_json_response(self, text: str) -> dict:
        """Robustly parse JSON from LLM response, handling truncation."""
        # First try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object from surrounding text
        match = re.search(r'\{.*', text, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Try to repair truncated JSON by closing open structures
            repaired = json_str
            # Close any open strings
            open_quotes = repaired.count('"') % 2
            if open_quotes:
                repaired += '"'
            # Close open arrays/objects
            open_brackets = repaired.count('[') - repaired.count(']')
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += ']' * max(open_brackets, 0)
            repaired += '}' * max(open_braces, 0)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
            
            # Last resort: extract what we can with regex
            script_match = re.search(r'"(?:narration_script|summary_script)"\s*:\s*"((?:[^"\\]|\\.)*)', text)
            script = script_match.group(1) if script_match else ""
            tone_match = re.search(r'"tone"\s*:\s*"([^"]*)"', text)
            tone = tone_match.group(1) if tone_match else "neutral"
            
            logger.warning(f"Repaired truncated JSON response (extracted {len(script)} chars of script)")
            return {
                "summary_script": script,
                "selected_panels": [],
                "tone": tone,
                "key_events": [],
                "characters": []
            }
        
        raise json.JSONDecodeError("No JSON object found in response", text, 0)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self._model_name.startswith("gemini"):
            from google.genai import types
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=8000,
                )
            )
            return response.text
        
        elif self._model_name.startswith("gpt"):
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a professional video editor creating short-form manga videos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {self._model_name}")
    
    def _placeholder_analyze(self, input_data: StoryIntelligenceInput) -> StoryIntelligenceOutput:
        """Perform placeholder analysis when no LLM is available."""
        logger.info("Using placeholder story analysis")
        
        # Combine all text
        all_text = " ".join([t for t in input_data.panel_texts if t.strip()])
        
        # Calculate target word count for proper duration
        target_words = int(input_data.max_duration_seconds * self.WORDS_PER_SECOND * 0.8)
        target_words = max(target_words, 100)  # At least 100 words for decent narration
        
        # Generate script
        if all_text and len(all_text.split()) > 20:
            # Use extracted text as base, clean it up
            words = all_text.split()
            script_words = words[:target_words]
            script = " ".join(script_words)
            
            # Add hook intro and payoff outro
            if self._anime_title:
                intro = f"What happens next in {self._anime_title} will leave you speechless. "
                outro = f" This is just the beginning of an incredible chapter in {self._anime_title}."
            else:
                intro = "You won't believe what happens next in this manga story. "
                outro = " The story continues with even more explosive developments ahead."
            script = intro + script + outro
        else:
            # Generate a generic narration for manga without text
            script = self._generate_generic_narration(
                len(input_data.panel_indices), 
                target_words,
                input_data.narration_style
            )
        
        # Select panels (simple strategy: evenly distributed)
        total_panels = len(input_data.panel_indices)
        max_panels = min(10, total_panels)
        
        if total_panels <= max_panels:
            selected = list(input_data.panel_indices)
        else:
            # Select evenly distributed panels, always include first and last
            selected = [input_data.panel_indices[0]]
            step = (total_panels - 1) / (max_panels - 1)
            for i in range(1, max_panels):
                idx = min(int(i * step), total_panels - 1)
                if input_data.panel_indices[idx] not in selected:
                    selected.append(input_data.panel_indices[idx])
            selected.sort()
        
        # Estimate duration
        word_count = len(script.split())
        estimated_duration = word_count / self.WORDS_PER_SECOND
        
        logger.info(f"Generated {word_count} word script, estimated {estimated_duration:.1f}s")
        
        return StoryIntelligenceOutput(
            summary_script=script,
            selected_panels=selected,
            tone="neutral",
            key_events=[],
            characters=[],
            estimated_duration=min(estimated_duration, input_data.max_duration_seconds),
            confidence=0.3,
            intro_hook=f"You won't believe what happens next in {self._anime_title or 'this manga'}...",
            metadata={
                'model': 'placeholder',
                'note': 'No LLM available',
                'intro_panel_index': input_data.panel_indices[0] if input_data.panel_indices else None,
            }
        )
    
    def _generate_generic_narration(
        self, 
        panel_count: int, 
        target_words: int,
        style: str = "engaging"
    ) -> str:
        """Generate a generic narration for manga without extracted text."""
        
        # Build narration based on panel count and style
        narration_parts = []
        
        # Opening
        openings = [
            "Welcome to this captivating manga story.",
            "Join us as we explore this visual narrative.",
            "This manga takes us on an incredible journey.",
            "Prepare to be immersed in this compelling tale.",
        ]
        narration_parts.append(openings[panel_count % len(openings)])
        
        # Middle content - describe the visual journey
        middle_parts = [
            "The story unfolds through beautifully crafted panels, each one revealing new details about our characters and their world.",
            "We follow the protagonist through a series of events that will change everything.",
            "The artwork captures intense emotions, from quiet moments of reflection to explosive action sequences.",
            "Each panel draws us deeper into the narrative, building tension and anticipation.",
            "The visual storytelling masterfully conveys the mood and atmosphere of each scene.",
            "Character expressions and dynamic compositions bring the story to life in vivid detail.",
            "The pacing moves from calm, introspective moments to heart-pounding action.",
            "Through light and shadow, the artist creates a world that feels both fantastical and real.",
            "The story builds momentum as we journey alongside the characters through their trials.",
            "Every frame is carefully composed to maximize emotional impact and narrative flow.",
        ]
        
        # Add middle parts until we reach target word count
        current_words = len(" ".join(narration_parts).split())
        part_idx = 0
        
        while current_words < target_words - 30 and part_idx < len(middle_parts):
            narration_parts.append(middle_parts[part_idx])
            current_words = len(" ".join(narration_parts).split())
            part_idx += 1
        
        # Add more generic content if needed
        while current_words < target_words - 30:
            filler = [
                "The characters face challenges that test their resolve and reveal their true nature.",
                "Subtle visual cues hint at deeper meanings beneath the surface of the story.",
                "The artistic style perfectly complements the emotional beats of the narrative.",
            ]
            narration_parts.append(filler[current_words % len(filler)])
            current_words = len(" ".join(narration_parts).split())
        
        # Closing
        closings = [
            "This is just the beginning of an unforgettable story.",
            "The journey continues with even more excitement ahead.",
            "Stay tuned for more from this amazing series.",
            "Thank you for joining us on this visual adventure.",
        ]
        narration_parts.append(closings[panel_count % len(closings)])
        
        return " ".join(narration_parts)
    def generate_script(
        self, 
        panel_texts: List[str], 
        max_words: int = 150,
        style: str = "engaging"
    ) -> str:
        """Generate a narration script from panel texts."""
        if self._model_name == "placeholder":
            return self._placeholder_script(panel_texts, max_words)
        
        prompt = f"""Create a {style} narration script for a short video.

Panel texts: {' | '.join(panel_texts)}

Requirements:
- Maximum {max_words} words
- {style} tone
- Suitable for video narration

Provide ONLY the narration text."""

        try:
            return self._call_llm(prompt).strip()
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return self._placeholder_script(panel_texts, max_words)
    
    def _placeholder_script(self, panel_texts: List[str], max_words: int) -> str:
        """Generate placeholder script."""
        combined = " ".join([t for t in panel_texts if t.strip()])
        if combined:
            words = combined.split()[:max_words]
            return " ".join(words)
        return "Explore the visual story unfolding panel by panel."
    
    def select_key_panels(
        self,
        panel_texts: List[str],
        max_panels: int = 10,
        selection_strategy: str = "llm"
    ) -> List[int]:
        """Select the most important panels for the video."""
        total = len(panel_texts)
        
        if selection_strategy == "llm" and self._model_name != "placeholder":
            try:
                return self._llm_select_panels(panel_texts, max_panels)
            except Exception as e:
                logger.error(f"LLM panel selection failed: {e}")
        
        # Heuristic selection
        return self._heuristic_select_panels(panel_texts, max_panels)
    
    def _llm_select_panels(self, panel_texts: List[str], max_panels: int) -> List[int]:
        """Select panels using LLM."""
        panel_list = "\n".join([
            f"Panel {i}: {text}" for i, text in enumerate(panel_texts)
        ])
        
        prompt = f"""Select the {max_panels} most important panels for a video.

Available panels:
{panel_list}

Return ONLY a JSON array of panel indices, e.g., [0, 2, 5, 8]"""

        response = self._call_llm(prompt)
        
        # Extract JSON array from response
        match = re.search(r'\[[\d,\s]+\]', response)
        if match:
            return json.loads(match.group())
        return list(range(min(max_panels, len(panel_texts))))
    
    def _heuristic_select_panels(self, panel_texts: List[str], max_panels: int) -> List[int]:
        """Select panels using heuristics."""
        total = len(panel_texts)
        
        if total <= max_panels:
            return list(range(total))
        
        # Score panels based on text length (more text = more important dialogue)
        scores = [(i, len(text)) for i, text in enumerate(panel_texts)]
        scores.sort(key=lambda x: -x[1])
        
        # Take top panels by text length
        selected = [idx for idx, _ in scores[:max_panels]]
        selected.sort()  # Maintain order
        
        return selected
    
    def detect_tone(self, panel_texts: List[str]) -> NarrativeTone:
        """Detect the narrative tone of the manga."""
        if self._model_name != "placeholder":
            try:
                return self._llm_detect_tone(panel_texts)
            except Exception:
                pass
        
        return NarrativeTone.NEUTRAL
    
    def _llm_detect_tone(self, panel_texts: List[str]) -> NarrativeTone:
        """Detect tone using LLM."""
        combined = " ".join(panel_texts)
        
        prompt = f"""Analyze this manga dialogue and determine the narrative tone.

Text: {combined[:1000]}

Respond with ONLY one of: dramatic, comedic, action, romantic, mysterious, horror, slice_of_life, neutral"""

        response = self._call_llm(prompt).strip().lower()
        
        tone_map = {
            "dramatic": NarrativeTone.DRAMATIC,
            "comedic": NarrativeTone.COMEDIC,
            "action": NarrativeTone.ACTION,
            "romantic": NarrativeTone.ROMANTIC,
            "mysterious": NarrativeTone.MYSTERIOUS,
            "horror": NarrativeTone.HORROR,
            "slice_of_life": NarrativeTone.SLICE_OF_LIFE,
        }
        
        return tone_map.get(response, NarrativeTone.NEUTRAL)
    
    def adjust_for_duration(
        self,
        script: str,
        target_duration: float,
        words_per_second: float = 2.5
    ) -> str:
        """Adjust script length to fit target duration."""
        max_words = int(target_duration * words_per_second)
        words = script.split()
        
        if len(words) <= max_words:
            return script
        
        # Truncate to max words
        truncated = words[:max_words]
        
        # Try to end at a sentence boundary
        result = " ".join(truncated)
        
        # Find last sentence end
        last_period = result.rfind('.')
        last_exclaim = result.rfind('!')
        last_question = result.rfind('?')
        
        last_end = max(last_period, last_exclaim, last_question)
        
        if last_end > len(result) * 0.7:  # Only truncate if we keep 70%+
            result = result[:last_end + 1]
        
        return result

    # ───────────────── AI Intro Image Generation ─────────────────

    def generate_intro_image(
        self,
        summary: str,
        tone: str = "dramatic",
        characters: Optional[List[str]] = None,
        manga_title: str = "",
        character_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        """Generate an AI intro image using Gemini's image generation.

        Creates a character-focused, coloured anime illustration designed to
        STOP SCROLLING — used as the title-card background.

        Strategy:
        1. If female characters exist → generate an attractive, eye-catching
           portrait of the most prominent one in a dramatic/beautiful pose.
        2. Otherwise → generate a powerful character shot of the protagonist
           or the most dramatic moment.
        3. Always character-focused, never a generic landscape.

        Args:
            summary: Story summary / narration script
            tone: Detected narrative tone
            characters: Character names mentioned in this chapter
            manga_title: Title of the manga
            character_details: List of dicts with name/gender/features/role

        Returns:
            BGR numpy array (portrait) or None on failure.
        """
        if not self._client or not self._model_name.startswith("gemini"):
            logger.info("Gemini client unavailable — skipping AI intro image")
            return None

        # ── Pick the focal character for the intro image ──
        focal_char = None
        focal_desc = ""
        char_details = character_details or []

        # Priority 1: Female characters (eye-catching for engagement)
        female_chars = [c for c in char_details if c.get("gender", "").lower() == "female"]
        if female_chars:
            focal_char = female_chars[0]
            features = focal_char.get("features", "beautiful anime girl")
            name = focal_char.get("name", "the heroine")
            focal_desc = (
                f"Focus on {name}: {features}. "
                f"She should be the clear subject of the image. "
                f"Make her look stunning — beautiful face, detailed eyes, "
                f"expressive pose (confident / fierce / elegant / mysterious, "
                f"whichever fits the tone: {tone}). "
                f"Draw her in a way that makes people stop scrolling."
            )
            logger.info(f"Intro image focal character (female): {name}")

        # Priority 2: Protagonist or first character
        if not focal_char and char_details:
            # Try protagonist first
            protag = [c for c in char_details if c.get("role", "").lower() == "protagonist"]
            focal_char = protag[0] if protag else char_details[0]
            features = focal_char.get("features", "anime character")
            name = focal_char.get("name", "the main character")
            focal_desc = (
                f"Focus on {name}: {features}. "
                f"Show them in a powerful, dramatic pose that conveys {tone} energy. "
                f"Make the character look badass / cool / intimidating as appropriate."
            )
            logger.info(f"Intro image focal character: {name}")

        # Priority 3: Just use character names if no details
        if not focal_char and characters:
            name = characters[0]
            focal_desc = (
                f"Focus on {name} from {manga_title or 'this anime'}. "
                f"Show them in a visually striking pose."
            )

        # Take the first ~80 words of the summary for scene context
        summary_short = " ".join(summary.split()[:80])

        prompt = (
            f"Generate a single HIGH-QUALITY anime character illustration. "
            f"This is a thumbnail/intro image for a YouTube Shorts video about "
            f"\"{manga_title or 'an anime series'}\".\n\n"
            f"SUBJECT: {focal_desc}\n\n"
            f"Scene mood: {tone}. Context: {summary_short}\n\n"
            f"STRICT REQUIREMENTS:\n"
            f"- CHARACTER must fill at least 60% of the frame (close-up or medium shot)\n"
            f"- Beautiful anime art style, high detail on face and eyes\n"
            f"- Vivid colours, dramatic/cinematic lighting\n"
            f"- Portrait orientation (9:16 ratio, taller than wide)\n"
            f"- Atmospheric background that matches the mood (blurred/bokeh OK)\n"
            f"- NO text, NO speech bubbles, NO watermarks, NO UI elements\n"
            f"- The image should make someone STOP scrolling — visually stunning\n"
            f"- Professional anime illustration quality (studio level)"
        )

        # Try multiple models in order of preference
        image_models = [
            "gemini-2.0-flash-exp-image-generation",
            "gemini-2.0-flash-preview-image-generation",
            "imagen-3.0-generate-002",
        ]

        from google.genai import types as genai_types

        for model_name in image_models:
            try:
                logger.info(f"Attempting intro image with {model_name}")
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=1.0,
                    ),
                )

                # Extract image bytes from the response
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            img_bytes = part.inline_data.data
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if img is not None:
                                logger.info(
                                    f"AI intro image generated via {model_name}: "
                                    f"{img.shape[1]}x{img.shape[0]}"
                                )
                                return img

                logger.info(f"{model_name}: no image in response, trying next")

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.info(f"{model_name}: quota exhausted, trying next")
                elif "not found" in err_str.lower() or "404" in err_str:
                    logger.info(f"{model_name}: model not available, trying next")
                else:
                    logger.warning(f"{model_name} failed: {err_str[:200]}")

        logger.warning("All image generation models exhausted — using fallback intro")
        return None
