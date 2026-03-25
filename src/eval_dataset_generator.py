"""
Evaluation Dataset Generator for RAG system.
Creates Q&A pairs from dog breed data for RAGAS evaluation.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import (
    BREED_MAPPING_FILE,
    EVAL_DIR,
    EVAL_DATASET_SIZE,
)

logger = logging.getLogger(__name__)


class EvalDatasetGenerator:
    """Generate evaluation Q&A pairs from dog breed data."""

    def __init__(self):
        """Initialize the generator."""
        # Load breed data
        with open(BREED_MAPPING_FILE, 'r') as f:
            data = json.load(f)
        self.breeds = data['breeds']
        logger.info(f"Loaded {len(self.breeds)} breeds for evaluation dataset")

        # Question templates for different types of queries
        self.question_templates = {
            'specific_breed': [
                "What are the characteristics of {breed}?",
                "Tell me about {breed}.",
                "What is the temperament of {breed}?",
                "What health problems are common in {breed}?",
                "Where does the {breed} originate from?",
                "What color is a {breed}?",
                "How long does a {breed} typically live?",
                "What is the average height of a {breed}?",
            ],
            'comparison': [
                "Compare {breed1} and {breed2}.",
                "What are the differences between {breed1} and {breed2}?",
                "Which is bigger, {breed1} or {breed2}?",
            ],
            'search': [
                "What dog breeds are from {country}?",
                "Which breeds have {color} fur?",
                "What are some {temperament} dog breeds?",
                "Which breeds are good for people with allergies?",
                "What large dog breeds have a friendly temperament?",
                "Which small breeds are energetic?",
            ],
        }

    def generate_specific_breed_qa(self, breed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Q&A pair about a specific breed.

        Args:
            breed_info: Breed information dictionary

        Returns:
            Q&A pair with ground truth
        """
        breed_name = breed_info['name']
        chars = breed_info['characteristics']

        # Choose random question template
        template = random.choice(self.question_templates['specific_breed'])
        question = template.format(breed=breed_name)

        # Generate answer based on question type
        if 'characteristics' in question.lower() or 'about' in question.lower():
            answer = (
                f"{breed_name} is a dog breed from {chars.get('Country of Origin', 'unknown')}. "
                f"They have {chars.get('Fur Color', 'various')} colored fur and typically stand "
                f"{chars.get('Height (in)', 'unknown')} inches tall. Their temperament is described as "
                f"{chars.get('Character Traits', 'unknown')}. They typically live "
                f"{chars.get('Longevity (yrs)', 'unknown')} years. "
                f"Common health problems include: {chars.get('Common Health Problems', 'unknown')}."
            )
        elif 'temperament' in question.lower():
            answer = f"{breed_name} are known to be {chars.get('Character Traits', 'unknown')}."
        elif 'health' in question.lower():
            answer = f"Common health problems in {breed_name} include: {chars.get('Common Health Problems', 'unknown')}."
        elif 'origin' in question.lower():
            answer = f"{breed_name} originated from {chars.get('Country of Origin', 'unknown')}."
        elif 'color' in question.lower():
            answer = f"{breed_name} typically have {chars.get('Fur Color', 'unknown')} colored fur."
        elif 'live' in question.lower() or 'lifespan' in question.lower():
            answer = f"{breed_name} typically live {chars.get('Longevity (yrs)', 'unknown')} years."
        elif 'height' in question.lower() or 'tall' in question.lower() or 'big' in question.lower():
            answer = f"{breed_name} typically stand {chars.get('Height (in)', 'unknown')} inches tall."
        else:
            answer = f"{breed_name} is from {chars.get('Country of Origin', 'unknown')} with {chars.get('Character Traits', 'unknown')} temperament."

        return {
            'question': question,
            'answer': answer,
            'ground_truth': answer,  # For RAGAS evaluation
            'contexts': [self._format_breed_context(breed_info)],
            'breed': breed_name,
            'type': 'specific_breed',
        }

    def generate_comparison_qa(self, breed1: Dict[str, Any], breed2: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Q&A comparing two breeds."""
        name1 = breed1['name']
        name2 = breed2['name']
        chars1 = breed1['characteristics']
        chars2 = breed2['characteristics']

        template = random.choice(self.question_templates['comparison'])
        question = template.format(breed1=name1, breed2=name2)

        # Compare key attributes
        answer = f"{name1} and {name2} are both popular dog breeds. "
        answer += f"{name1} is from {chars1.get('Country of Origin', 'unknown')}, while {name2} is from {chars2.get('Country of Origin', 'unknown')}. "
        answer += f"{name1} stands {chars1.get('Height (in)', 'unknown')} inches tall, compared to {name2}'s {chars2.get('Height (in)', 'unknown')} inches. "
        answer += f"In terms of temperament, {name1} is {chars1.get('Character Traits', 'unknown')}, while {name2} is {chars2.get('Character Traits', 'unknown')}."

        return {
            'question': question,
            'answer': answer,
            'ground_truth': answer,
            'contexts': [
                self._format_breed_context(breed1),
                self._format_breed_context(breed2),
            ],
            'breeds': [name1, name2],
            'type': 'comparison',
        }

    def generate_search_qa(self) -> Dict[str, Any]:
        """Generate Q&A for search-type questions."""
        # Pick a random search criteria
        search_type = random.choice(['country', 'color', 'temperament'])

        if search_type == 'country':
            # Find breeds from same country
            countries = [b['characteristics'].get('Country of Origin') for b in self.breeds]
            country = random.choice([c for c in countries if c])

            matching_breeds = [
                b for b in self.breeds
                if b['characteristics'].get('Country of Origin') == country
            ]

            question = f"What dog breeds are from {country}?"
            breed_names = [b['name'] for b in matching_breeds[:5]]
            answer = f"Dog breeds from {country} include: {', '.join(breed_names)}."

        elif search_type == 'color':
            # Find breeds with specific color
            all_colors = set()
            for b in self.breeds:
                colors = b['characteristics'].get('Fur Color', '').split(', ')
                all_colors.update(colors)

            color = random.choice(list(all_colors))
            matching_breeds = [
                b for b in self.breeds
                if color in b['characteristics'].get('Fur Color', '')
            ]

            question = f"Which dog breeds have {color} fur?"
            breed_names = [b['name'] for b in matching_breeds[:5]]
            answer = f"Dog breeds with {color} fur include: {', '.join(breed_names)}."

        else:  # temperament
            # Search for friendly breeds
            question = "What dog breeds are good with children?"
            matching_breeds = [
                b for b in self.breeds
                if 'friendly' in b['characteristics'].get('Character Traits', '').lower()
                or 'good-natured' in b['characteristics'].get('Character Traits', '').lower()
            ]
            breed_names = [b['name'] for b in matching_breeds[:5]]
            answer = f"Dog breeds that are good with children (friendly, good-natured) include: {', '.join(breed_names)}."

        contexts = [self._format_breed_context(b) for b in matching_breeds[:3]]

        return {
            'question': question,
            'answer': answer,
            'ground_truth': answer,
            'contexts': contexts,
            'type': 'search',
            'search_criteria': search_type,
        }

    def _format_breed_context(self, breed_info: Dict[str, Any]) -> str:
        """Format breed information as context string."""
        chars = breed_info['characteristics']
        context = f"Breed: {breed_info['name']}\n"
        for key, value in chars.items():
            context += f"{key}: {value}\n"
        return context

    def generate_dataset(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """
        Generate complete evaluation dataset.

        Args:
            num_samples: Number of Q&A pairs to generate

        Returns:
            List of Q&A pairs
        """
        if num_samples is None:
            num_samples = EVAL_DATASET_SIZE

        dataset = []

        # Generate different types of questions
        num_specific = int(num_samples * 0.6)  # 60% specific breed questions
        num_comparison = int(num_samples * 0.2)  # 20% comparison questions
        num_search = num_samples - num_specific - num_comparison  # 20% search questions

        logger.info(f"Generating {num_specific} specific breed questions...")
        for _ in range(num_specific):
            breed = random.choice(self.breeds)
            qa = self.generate_specific_breed_qa(breed)
            dataset.append(qa)

        logger.info(f"Generating {num_comparison} comparison questions...")
        for _ in range(num_comparison):
            breed1, breed2 = random.sample(self.breeds, 2)
            qa = self.generate_comparison_qa(breed1, breed2)
            dataset.append(qa)

        logger.info(f"Generating {num_search} search questions...")
        for _ in range(num_search):
            qa = self.generate_search_qa()
            dataset.append(qa)

        # Shuffle dataset
        random.shuffle(dataset)

        logger.info(f"✓ Generated {len(dataset)} Q&A pairs")
        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "eval_dataset.json"):
        """Save dataset to file."""
        filepath = EVAL_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved evaluation dataset to {filepath}")
        return filepath

    def load_dataset(self, filename: str = "eval_dataset.json") -> List[Dict[str, Any]]:
        """Load dataset from file."""
        filepath = EVAL_DIR / filename

        if not filepath.exists():
            logger.warning(f"Dataset file not found: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        logger.info(f"✓ Loaded {len(dataset)} Q&A pairs from {filepath}")
        return dataset


def main():
    """Generate and save evaluation dataset."""
    print("=" * 60)
    print("Evaluation Dataset Generator")
    print("=" * 60)

    generator = EvalDatasetGenerator()

    # Generate dataset
    print(f"\nGenerating {EVAL_DATASET_SIZE} Q&A pairs...")
    dataset = generator.generate_dataset(num_samples=EVAL_DATASET_SIZE)

    # Show samples
    print("\nSample Q&A pairs:")
    print("-" * 60)
    for i, qa in enumerate(dataset[:3], 1):
        print(f"\n{i}. Type: {qa['type']}")
        print(f"   Question: {qa['question']}")
        print(f"   Answer: {qa['answer'][:150]}...")

    # Save dataset
    print("\nSaving dataset...")
    filepath = generator.save_dataset(dataset)
    print(f"✓ Saved to: {filepath}")

    # Statistics
    print("\nDataset Statistics:")
    print("-" * 60)
    types = {}
    for qa in dataset:
        types[qa['type']] = types.get(qa['type'], 0) + 1

    for qtype, count in types.items():
        print(f"  {qtype}: {count} questions ({count/len(dataset)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Evaluation dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
