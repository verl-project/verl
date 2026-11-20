from abc import ABC, abstractmethod

from bs4 import BeautifulSoup, Tag  # Ensure Tag is imported


class AccessibilityRule(ABC):
    """Abstract base class for an accessibility rule checker."""

    @property
    @abstractmethod
    def issue_key(self) -> str:
        """A unique string key identifying the type of issue this rule checks for.
        This should match the keys used in the 'issues_to_fix' list in your dataset.
        """
        pass

    @abstractmethod
    def check(self, soup: BeautifulSoup, original_html_info: dict) -> bool:
        """
        Checks the provided HTML (parsed as a BeautifulSoup soup) for compliance
        with this specific accessibility rule.

        Args:
            soup: The BeautifulSoup object representing the LLM's modified HTML.
            original_html_info: A dictionary containing information about the original
                                HTML snippet, potentially including the original HTML string
                                if needed for context by some rules.

        Returns:
            True if the HTML passes this rule (i.e., the issue is fixed or wasn't present).
            False if the HTML fails this rule (i.e., the issue persists or was introduced).
        """
        pass


class MissingAltTextRule(AccessibilityRule):
    @property
    def issue_key(self) -> str:
        return "missing_alt_text"

    def check(self, soup: BeautifulSoup, original_html_info: dict) -> bool:
        # Check if images were relevant in the first place for this item based on original_html_info
        # This helps decide if an absence of images now is a failure or if the check is moot.
        original_had_images = "<img" in original_html_info.get("html", "").lower()

        img_tags = soup.find_all("img")

        if original_had_images and not img_tags:
            # Relevant images were present in original, but LLM removed all img tags.
            return False

        if not original_had_images and not img_tags:
            # No images in original, no images in LLM output. Rule passes by default for this item.
            return True

        if (
            not img_tags and original_had_images
        ):  # Should be caught by the first check, but for clarity
            return False

        # If there are image tags, all must have valid alt text
        for img_element in img_tags:
            if isinstance(img_element, Tag):
                alt_value = img_element.get("alt")
                if not (isinstance(alt_value, str) and alt_value.strip() != ""):
                    return False  # Found an image without a valid alt attribute
            else:
                # This case should ideally not happen if soup.find_all('img') works as expected
                return False
        return True  # All images found have valid alt text


class LabelAssociationRule(AccessibilityRule):
    @property
    def issue_key(self) -> str:
        return "missing_label_for"  # Or "label_association" if you prefer

    def check(self, soup: BeautifulSoup, original_html_info: dict) -> bool:
        original_had_labels = "<label" in original_html_info.get("html", "").lower()

        label_tags = soup.find_all("label")

        if original_had_labels and not label_tags:
            return False

        if not original_had_labels and not label_tags:
            return True  # Rule passes by default

        if not label_tags and original_had_labels:
            return False

        for label_element in label_tags:
            if isinstance(label_element, Tag):
                for_value = label_element.get("for")
                has_explicit_for_association = False

                if isinstance(for_value, str):
                    for_attr_str = for_value.strip()
                    if for_attr_str != "":
                        # Check if an element with this ID exists and is an appropriate input type
                        target_element = soup.find(id=for_attr_str)
                        if target_element and target_element.name in [
                            "input",
                            "textarea",
                            "select",
                            "button",
                            "meter",
                            "output",
                            "progress",
                            "selectlist",
                        ]:
                            has_explicit_for_association = True

                # Check for implicit association (label contains the input element)
                # Note: This is a simpler check. True implicit association has more nuance.
                contains_input_directly = False
                # Only direct children that are form controls for simplicity here
                for child in label_element.children:
                    if isinstance(child, Tag) and child.name in [
                        "input",
                        "textarea",
                        "select",
                        "button",
                    ]:  # etc.
                        contains_input_directly = True
                        break

                if not (has_explicit_for_association or contains_input_directly):
                    return False  # Found a label not correctly associated
            else:
                return False
        return True  # All labels are correctly associated


# --- Add more rule classes here as needed ---
# e.g., class AriaAttributeRule(AccessibilityRule): ...
# e.g., class ContrastRule(AccessibilityRule): ...
