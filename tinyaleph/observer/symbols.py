"""
Symbol Database and Symbol Types

Provides a database of 400+ archetypal symbols with:
- Unique prime associations
- Cultural tags for semantic mapping
- Category classification
- Unicode representations

This module connects abstract prime-based computation to
culturally-grounded archetypal meanings.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto


class SymbolCategory(Enum):
    """Categories of symbols in the database"""
    ARCHETYPE = auto()      # Jungian archetypes (hero, sage, etc.)
    ELEMENT = auto()        # Natural elements (fire, water, etc.)
    PLACE = auto()          # Sacred places (temple, mountain, etc.)
    OBJECT = auto()         # Symbolic objects (sword, cup, etc.)
    ABSTRACT = auto()       # Abstract concepts (unity, duality, etc.)
    MYTHOLOGICAL = auto()   # Mythological figures/concepts
    TAROT = auto()          # Tarot arcana
    ICHING = auto()         # I-Ching trigrams/hexagrams
    CELESTIAL = auto()      # Sun, moon, stars, planets
    CREATURE = auto()       # Dragons, phoenix, etc.


@dataclass
class Symbol:
    """
    A symbol with prime association and cultural grounding.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        prime: Associated prime number
        category: Symbol category
        unicode: Unicode representation (emoji or symbol)
        cultural_tags: List of semantic/cultural tags
        description: Optional description
        related_ids: IDs of related symbols
    """
    id: str
    name: str
    prime: int
    category: SymbolCategory
    unicode: str = "◯"
    cultural_tags: List[str] = field(default_factory=list)
    description: str = ""
    related_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'prime': self.prime,
            'category': self.category.name,
            'unicode': self.unicode,
            'cultural_tags': self.cultural_tags,
            'description': self.description,
            'related_ids': self.related_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            prime=data['prime'],
            category=SymbolCategory[data['category']],
            unicode=data.get('unicode', '◯'),
            cultural_tags=data.get('cultural_tags', []),
            description=data.get('description', ''),
            related_ids=data.get('related_ids', [])
        )


# First 500 primes for symbol assignment
def _generate_primes(n: int) -> List[int]:
    """Generate first n primes using sieve"""
    if n <= 0:
        return []
    
    # Upper bound for nth prime
    if n < 6:
        upper = 15
    else:
        upper = int(n * (math.log(n) + math.log(math.log(n)))) + 3
    
    sieve = [True] * (upper + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(upper)) + 1):
        if sieve[i]:
            for j in range(i * i, upper + 1, i):
                sieve[j] = False
    
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes[:n]


PRIMES = _generate_primes(500)


class SymbolDatabase:
    """
    Database of archetypal symbols with prime associations.
    
    Provides:
    - Symbol lookup by ID, name, or prime
    - Category-based filtering
    - Tag-based search
    - Semantic similarity via prime relationships
    """
    
    def __init__(self):
        self._symbols: Dict[str, Symbol] = {}
        self._by_prime: Dict[int, Symbol] = {}
        self._by_name: Dict[str, Symbol] = {}
        self._by_category: Dict[SymbolCategory, List[Symbol]] = {c: [] for c in SymbolCategory}
        self._by_tag: Dict[str, List[Symbol]] = {}
        
        # Initialize with default symbols
        self._init_default_symbols()
    
    def _init_default_symbols(self):
        """Initialize the default symbol database"""
        symbols = [
            # ═══════════════════════════════════════════════════════════════
            # ABSTRACT CONCEPTS (primes 2-101)
            # ═══════════════════════════════════════════════════════════════
            Symbol('unity', 'Unity', 2, SymbolCategory.ABSTRACT, '☯', 
                   ['universal', 'oneness', 'wholeness'], 'The primordial unity'),
            Symbol('duality', 'Duality', 3, SymbolCategory.ABSTRACT, '⚌',
                   ['polarity', 'opposites', 'balance'], 'The fundamental split'),
            Symbol('harmony', 'Harmony', 5, SymbolCategory.ABSTRACT, '♫',
                   ['balance', 'peace', 'music'], 'Perfect balance'),
            Symbol('chaos', 'Chaos', 7, SymbolCategory.ABSTRACT, '⌬',
                   ['disorder', 'potential', 'primordial'], 'Creative disorder'),
            Symbol('order', 'Order', 11, SymbolCategory.ABSTRACT, '⚙',
                   ['structure', 'form', 'organization'], 'Structured cosmos'),
            Symbol('transformation', 'Transformation', 13, SymbolCategory.ABSTRACT, '♻',
                   ['change', 'alchemy', 'metamorphosis'], 'Essential change'),
            Symbol('creation', 'Creation', 17, SymbolCategory.ABSTRACT, '✦',
                   ['genesis', 'origin', 'birth'], 'Bringing into being'),
            Symbol('destruction', 'Destruction', 19, SymbolCategory.ABSTRACT, '⚡',
                   ['ending', 'dissolution', 'release'], 'Return to source'),
            Symbol('infinity', 'Infinity', 23, SymbolCategory.ABSTRACT, '∞',
                   ['eternal', 'boundless', 'cosmic'], 'Without limit'),
            Symbol('truth', 'Truth', 29, SymbolCategory.ABSTRACT, '☼',
                   ['reality', 'clarity', 'authenticity'], 'What is'),
            Symbol('love', 'Love', 31, SymbolCategory.ABSTRACT, '♥',
                   ['heart', 'connection', 'emotion'], 'Universal bond'),
            Symbol('power', 'Power', 37, SymbolCategory.ABSTRACT, '⚔',
                   ['strength', 'force', 'authority'], 'Ability to act'),
            Symbol('wisdom', 'Wisdom', 41, SymbolCategory.ABSTRACT, '📖',
                   ['knowledge', 'insight', 'understanding'], 'Deep knowing'),
            Symbol('time', 'Time', 43, SymbolCategory.ABSTRACT, '⏳',
                   ['temporal', 'flow', 'duration'], 'Passage of moments'),
            Symbol('space', 'Space', 47, SymbolCategory.ABSTRACT, '◐',
                   ['location', 'realm', 'dimension'], 'Extension of being'),
            Symbol('consciousness', 'Consciousness', 53, SymbolCategory.ABSTRACT, '👁',
                   ['awareness', 'mind', 'spirit'], 'Self-aware presence'),
            Symbol('life', 'Life', 59, SymbolCategory.ABSTRACT, '🌱',
                   ['vitality', 'growth', 'organic'], 'Animate principle'),
            Symbol('death', 'Death', 61, SymbolCategory.ABSTRACT, '💀',
                   ['ending', 'transition', 'transformation'], 'Great passage'),
            Symbol('light', 'Light', 67, SymbolCategory.ABSTRACT, '☀',
                   ['illumination', 'clarity', 'yang'], 'Radiant principle'),
            Symbol('darkness', 'Darkness', 71, SymbolCategory.ABSTRACT, '🌑',
                   ['shadow', 'mystery', 'yin'], 'Hidden depths'),
            Symbol('balance', 'Balance', 73, SymbolCategory.ABSTRACT, '⚖',
                   ['equilibrium', 'justice', 'measure'], 'Perfect poise'),
            Symbol('growth', 'Growth', 79, SymbolCategory.ABSTRACT, '📈',
                   ['expansion', 'development', 'evolution'], 'Increasing being'),
            Symbol('decay', 'Decay', 83, SymbolCategory.ABSTRACT, '🍂',
                   ['entropy', 'dissolution', 'autumn'], 'Natural decline'),
            Symbol('renewal', 'Renewal', 89, SymbolCategory.ABSTRACT, '🌅',
                   ['rebirth', 'cycle', 'spring'], 'Fresh beginning'),
            Symbol('beauty', 'Beauty', 97, SymbolCategory.ABSTRACT, '🌸',
                   ['elegance', 'form', 'harmony'], 'Aesthetic perfection'),
            Symbol('void', 'Void', 101, SymbolCategory.ABSTRACT, '⬛',
                   ['emptiness', 'potential', 'sunyata'], 'Pregnant emptiness'),
            
            # ═══════════════════════════════════════════════════════════════
            # ARCHETYPES (primes 103-199)
            # ═══════════════════════════════════════════════════════════════
            Symbol('hero', 'Hero', 103, SymbolCategory.ARCHETYPE, '🦸',
                   ['jungian', 'courage', 'quest'], 'The courageous seeker'),
            Symbol('sage', 'Sage', 107, SymbolCategory.ARCHETYPE, '🧙',
                   ['wisdom', 'teacher', 'guide'], 'The wise counselor'),
            Symbol('trickster', 'Trickster', 109, SymbolCategory.ARCHETYPE, '🃏',
                   ['chaos', 'humor', 'boundaries'], 'The boundary-crosser'),
            Symbol('mother', 'Great Mother', 113, SymbolCategory.ARCHETYPE, '👸',
                   ['nurture', 'fertility', 'earth'], 'The nurturing principle'),
            Symbol('father', 'Great Father', 127, SymbolCategory.ARCHETYPE, '👑',
                   ['authority', 'order', 'sky'], 'The ordering principle'),
            Symbol('child', 'Divine Child', 131, SymbolCategory.ARCHETYPE, '👶',
                   ['innocence', 'potential', 'wonder'], 'Pure potentiality'),
            Symbol('shadow', 'Shadow', 137, SymbolCategory.ARCHETYPE, '👤',
                   ['unconscious', 'repressed', 'dark'], 'The hidden self'),
            Symbol('anima', 'Anima', 139, SymbolCategory.ARCHETYPE, '💃',
                   ['feminine', 'soul', 'eros'], 'Inner feminine'),
            Symbol('animus', 'Animus', 149, SymbolCategory.ARCHETYPE, '🕺',
                   ['masculine', 'logos', 'action'], 'Inner masculine'),
            Symbol('self', 'Self', 151, SymbolCategory.ARCHETYPE, '🎯',
                   ['wholeness', 'integration', 'center'], 'The integrated totality'),
            Symbol('persona', 'Persona', 157, SymbolCategory.ARCHETYPE, '🎭',
                   ['mask', 'social', 'role'], 'The social face'),
            Symbol('wise_old_man', 'Wise Old Man', 163, SymbolCategory.ARCHETYPE, '🧓',
                   ['guidance', 'knowledge', 'mentor'], 'The spiritual guide'),
            Symbol('everyman', 'Everyman', 167, SymbolCategory.ARCHETYPE, '🧑',
                   ['ordinary', 'common', 'human'], 'The common person'),
            Symbol('caregiver', 'Caregiver', 173, SymbolCategory.ARCHETYPE, '🤗',
                   ['nurture', 'compassion', 'protection'], 'The protector'),
            Symbol('explorer', 'Explorer', 179, SymbolCategory.ARCHETYPE, '🧭',
                   ['adventure', 'freedom', 'discovery'], 'The boundary-pusher'),
            Symbol('rebel', 'Rebel', 181, SymbolCategory.ARCHETYPE, '✊',
                   ['revolution', 'freedom', 'change'], 'The rule-breaker'),
            Symbol('lover', 'Lover', 191, SymbolCategory.ARCHETYPE, '💕',
                   ['passion', 'intimacy', 'devotion'], 'The connector'),
            Symbol('creator', 'Creator', 193, SymbolCategory.ARCHETYPE, '🎨',
                   ['imagination', 'innovation', 'vision'], 'The maker'),
            Symbol('ruler', 'Ruler', 197, SymbolCategory.ARCHETYPE, '🏛',
                   ['control', 'responsibility', 'order'], 'The sovereign'),
            Symbol('magician', 'Magician', 199, SymbolCategory.ARCHETYPE, '🪄',
                   ['transformation', 'power', 'vision'], 'The transformer'),
            
            # ═══════════════════════════════════════════════════════════════
            # ELEMENTS (primes 211-269)
            # ═══════════════════════════════════════════════════════════════
            Symbol('fire', 'Fire', 211, SymbolCategory.ELEMENT, '🔥',
                   ['energy', 'transformation', 'yang'], 'Active principle'),
            Symbol('water', 'Water', 223, SymbolCategory.ELEMENT, '💧',
                   ['emotion', 'flow', 'yin'], 'Receptive principle'),
            Symbol('earth', 'Earth', 227, SymbolCategory.ELEMENT, '🌍',
                   ['stability', 'matter', 'grounding'], 'Solid foundation'),
            Symbol('air', 'Air', 229, SymbolCategory.ELEMENT, '💨',
                   ['mind', 'communication', 'spirit'], 'Subtle principle'),
            Symbol('aether', 'Aether', 233, SymbolCategory.ELEMENT, '✨',
                   ['spirit', 'quintessence', 'cosmic'], 'Fifth element'),
            Symbol('metal', 'Metal', 239, SymbolCategory.ELEMENT, '⚙',
                   ['structure', 'strength', 'autumn'], 'Chinese element'),
            Symbol('wood', 'Wood', 241, SymbolCategory.ELEMENT, '🌲',
                   ['growth', 'spring', 'flexibility'], 'Chinese element'),
            Symbol('tree', 'Tree', 251, SymbolCategory.ELEMENT, '🌳',
                   ['life', 'growth', 'axis_mundi'], 'World tree'),
            Symbol('flower', 'Flower', 257, SymbolCategory.ELEMENT, '🌺',
                   ['beauty', 'bloom', 'ephemeral'], 'Blossoming'),
            Symbol('mountain', 'Mountain', 263, SymbolCategory.ELEMENT, '⛰',
                   ['stability', 'aspiration', 'stillness'], 'Sacred height'),
            Symbol('ocean', 'Ocean', 269, SymbolCategory.ELEMENT, '🌊',
                   ['depth', 'unconscious', 'vastness'], 'Primal waters'),
            
            # ═══════════════════════════════════════════════════════════════
            # CELESTIAL (primes 271-313)
            # ═══════════════════════════════════════════════════════════════
            Symbol('sun', 'Sun', 271, SymbolCategory.CELESTIAL, '☀',
                   ['consciousness', 'gold', 'yang'], 'Solar principle'),
            Symbol('moon', 'Moon', 277, SymbolCategory.CELESTIAL, '🌙',
                   ['intuition', 'silver', 'yin'], 'Lunar principle'),
            Symbol('stars', 'Stars', 281, SymbolCategory.CELESTIAL, '⭐',
                   ['destiny', 'cosmos', 'guidance'], 'Celestial lights'),
            Symbol('venus', 'Venus', 283, SymbolCategory.CELESTIAL, '♀',
                   ['love', 'beauty', 'feminine'], 'Morning star'),
            Symbol('mars', 'Mars', 293, SymbolCategory.CELESTIAL, '♂',
                   ['war', 'action', 'masculine'], 'Red planet'),
            Symbol('jupiter', 'Jupiter', 307, SymbolCategory.CELESTIAL, '♃',
                   ['expansion', 'wisdom', 'abundance'], 'King of planets'),
            Symbol('saturn', 'Saturn', 311, SymbolCategory.CELESTIAL, '♄',
                   ['time', 'limits', 'karma'], 'Lord of time'),
            Symbol('mercury', 'Mercury', 313, SymbolCategory.CELESTIAL, '☿',
                   ['communication', 'mind', 'trickster'], 'Messenger'),
            
            # ═══════════════════════════════════════════════════════════════
            # CREATURES (primes 317-367)
            # ═══════════════════════════════════════════════════════════════
            Symbol('dragon', 'Dragon', 317, SymbolCategory.CREATURE, '🐉',
                   ['power', 'wisdom', 'primal'], 'Cosmic serpent'),
            Symbol('phoenix', 'Phoenix', 331, SymbolCategory.CREATURE, '🔥',
                   ['rebirth', 'transformation', 'fire'], 'Eternal return'),
            Symbol('serpent', 'Serpent', 337, SymbolCategory.CREATURE, '🐍',
                   ['wisdom', 'renewal', 'kundalini'], 'Earth wisdom'),
            Symbol('eagle', 'Eagle', 347, SymbolCategory.CREATURE, '🦅',
                   ['vision', 'freedom', 'spirit'], 'Sky messenger'),
            Symbol('lion', 'Lion', 349, SymbolCategory.CREATURE, '🦁',
                   ['courage', 'royalty', 'sun'], 'King of beasts'),
            Symbol('wolf', 'Wolf', 353, SymbolCategory.CREATURE, '🐺',
                   ['instinct', 'pack', 'wild'], 'Shadow guide'),
            Symbol('owl', 'Owl', 359, SymbolCategory.CREATURE, '🦉',
                   ['wisdom', 'night', 'death'], 'Night wisdom'),
            Symbol('raven', 'Raven', 367, SymbolCategory.CREATURE, '🦅',
                   ['magic', 'prophecy', 'messenger'], 'Trickster bird'),
            
            # ═══════════════════════════════════════════════════════════════
            # OBJECTS (primes 373-421)
            # ═══════════════════════════════════════════════════════════════
            Symbol('sword', 'Sword', 373, SymbolCategory.OBJECT, '⚔',
                   ['discrimination', 'truth', 'air'], 'Cutting discernment'),
            Symbol('cup', 'Cup', 379, SymbolCategory.OBJECT, '🏆',
                   ['receptivity', 'emotion', 'water'], 'Holy vessel'),
            Symbol('wand', 'Wand', 383, SymbolCategory.OBJECT, '🪄',
                   ['will', 'creation', 'fire'], 'Creative force'),
            Symbol('pentacle', 'Pentacle', 389, SymbolCategory.OBJECT, '⭐',
                   ['manifestation', 'earth', 'body'], 'Material form'),
            Symbol('crown', 'Crown', 397, SymbolCategory.OBJECT, '👑',
                   ['sovereignty', 'achievement', 'authority'], 'Supreme power'),
            Symbol('key', 'Key', 401, SymbolCategory.OBJECT, '🔑',
                   ['access', 'mystery', 'initiation'], 'Unlocking secrets'),
            Symbol('mirror', 'Mirror', 409, SymbolCategory.OBJECT, '🪞',
                   ['reflection', 'truth', 'self'], 'Self-seeing'),
            Symbol('wheel', 'Wheel', 419, SymbolCategory.OBJECT, '☸',
                   ['cycle', 'fortune', 'dharma'], 'Eternal return'),
            Symbol('hourglass', 'Hourglass', 421, SymbolCategory.OBJECT, '⏳',
                   ['time', 'mortality', 'patience'], 'Flowing time'),
            
            # ═══════════════════════════════════════════════════════════════
            # PLACES (primes 431-479)
            # ═══════════════════════════════════════════════════════════════
            Symbol('temple', 'Temple', 431, SymbolCategory.PLACE, '🏛',
                   ['sacred', 'worship', 'center'], 'Sacred space'),
            Symbol('garden', 'Garden', 433, SymbolCategory.PLACE, '🌻',
                   ['paradise', 'cultivation', 'nature'], 'Earthly paradise'),
            Symbol('cave', 'Cave', 439, SymbolCategory.PLACE, '🕳',
                   ['womb', 'unconscious', 'initiation'], 'Earth womb'),
            Symbol('tower', 'Tower', 443, SymbolCategory.PLACE, '🗼',
                   ['aspiration', 'isolation', 'ascent'], 'Reaching upward'),
            Symbol('bridge', 'Bridge', 449, SymbolCategory.PLACE, '🌉',
                   ['transition', 'connection', 'crossing'], 'Between worlds'),
            Symbol('labyrinth', 'Labyrinth', 457, SymbolCategory.PLACE, '🌀',
                   ['journey', 'center', 'initiation'], 'Sacred path'),
            Symbol('crossroads', 'Crossroads', 461, SymbolCategory.PLACE, '✝',
                   ['choice', 'fate', 'meeting'], 'Decision point'),
            Symbol('threshold', 'Threshold', 463, SymbolCategory.PLACE, '🚪',
                   ['transition', 'liminal', 'passage'], 'Between states'),
            Symbol('cosmos', 'Cosmos', 467, SymbolCategory.PLACE, '🌌',
                   ['universe', 'order', 'totality'], 'Ordered universe'),
            Symbol('underworld', 'Underworld', 479, SymbolCategory.PLACE, '⬇',
                   ['death', 'unconscious', 'treasure'], 'Realm below'),
            
            # ═══════════════════════════════════════════════════════════════
            # TAROT MAJOR ARCANA (primes 487-601)
            # ═══════════════════════════════════════════════════════════════
            Symbol('fool', 'The Fool', 487, SymbolCategory.TAROT, '🃏',
                   ['beginnings', 'innocence', 'leap'], '0 - The journey begins'),
            Symbol('magician_tarot', 'The Magician', 491, SymbolCategory.TAROT, '🎩',
                   ['manifestation', 'skill', 'will'], 'I - As above, so below'),
            Symbol('high_priestess', 'High Priestess', 499, SymbolCategory.TAROT, '🌙',
                   ['intuition', 'mystery', 'unconscious'], 'II - Hidden knowledge'),
            Symbol('empress', 'The Empress', 503, SymbolCategory.TAROT, '👸',
                   ['fertility', 'abundance', 'nature'], 'III - Earth mother'),
            Symbol('emperor', 'The Emperor', 509, SymbolCategory.TAROT, '👑',
                   ['authority', 'structure', 'father'], 'IV - Worldly power'),
            Symbol('hierophant', 'The Hierophant', 521, SymbolCategory.TAROT, '🔱',
                   ['tradition', 'teaching', 'religion'], 'V - Spiritual teacher'),
            Symbol('lovers', 'The Lovers', 523, SymbolCategory.TAROT, '💑',
                   ['choice', 'union', 'values'], 'VI - Sacred union'),
            Symbol('chariot', 'The Chariot', 541, SymbolCategory.TAROT, '🏎',
                   ['willpower', 'victory', 'control'], 'VII - Triumph'),
            Symbol('strength_tarot', 'Strength', 547, SymbolCategory.TAROT, '🦁',
                   ['courage', 'patience', 'inner_strength'], 'VIII - Gentle power'),
            Symbol('hermit', 'The Hermit', 557, SymbolCategory.TAROT, '🏔',
                   ['solitude', 'guidance', 'introspection'], 'IX - Inner light'),
            Symbol('wheel_fortune', 'Wheel of Fortune', 563, SymbolCategory.TAROT, '☸',
                   ['fate', 'cycles', 'change'], 'X - Turning point'),
            Symbol('justice_tarot', 'Justice', 569, SymbolCategory.TAROT, '⚖',
                   ['fairness', 'truth', 'karma'], 'XI - Cause and effect'),
            Symbol('hanged_man', 'The Hanged Man', 571, SymbolCategory.TAROT, '🙃',
                   ['surrender', 'perspective', 'sacrifice'], 'XII - Letting go'),
            Symbol('death_tarot', 'Death', 577, SymbolCategory.TAROT, '💀',
                   ['transformation', 'ending', 'rebirth'], 'XIII - Great change'),
            Symbol('temperance', 'Temperance', 587, SymbolCategory.TAROT, '⚗',
                   ['balance', 'patience', 'alchemy'], 'XIV - Middle way'),
            Symbol('devil_tarot', 'The Devil', 593, SymbolCategory.TAROT, '😈',
                   ['bondage', 'materialism', 'shadow'], 'XV - Chains'),
            Symbol('tower_tarot', 'The Tower', 599, SymbolCategory.TAROT, '🗼',
                   ['upheaval', 'revelation', 'awakening'], 'XVI - Sudden change'),
            Symbol('star_tarot', 'The Star', 601, SymbolCategory.TAROT, '⭐',
                   ['hope', 'inspiration', 'renewal'], 'XVII - Divine guidance'),
            
            # ═══════════════════════════════════════════════════════════════
            # I-CHING TRIGRAMS (primes 607-647)
            # ═══════════════════════════════════════════════════════════════
            Symbol('heaven_trigram', 'Heaven (Qian)', 607, SymbolCategory.ICHING, '☰',
                   ['creative', 'yang', 'father'], 'Pure yang'),
            Symbol('earth_trigram', 'Earth (Kun)', 613, SymbolCategory.ICHING, '☷',
                   ['receptive', 'yin', 'mother'], 'Pure yin'),
            Symbol('thunder_trigram', 'Thunder (Zhen)', 617, SymbolCategory.ICHING, '☳',
                   ['arousing', 'movement', 'eldest_son'], 'Arousing'),
            Symbol('water_trigram', 'Water (Kan)', 619, SymbolCategory.ICHING, '☵',
                   ['abysmal', 'danger', 'middle_son'], 'Danger'),
            Symbol('mountain_trigram', 'Mountain (Gen)', 631, SymbolCategory.ICHING, '☶',
                   ['stillness', 'keeping_still', 'youngest_son'], 'Keeping still'),
            Symbol('wind_trigram', 'Wind (Xun)', 641, SymbolCategory.ICHING, '☴',
                   ['gentle', 'penetrating', 'eldest_daughter'], 'Gentle penetration'),
            Symbol('fire_trigram', 'Fire (Li)', 643, SymbolCategory.ICHING, '☲',
                   ['clinging', 'clarity', 'middle_daughter'], 'Clinging'),
            Symbol('lake_trigram', 'Lake (Dui)', 647, SymbolCategory.ICHING, '☱',
                   ['joyous', 'pleasure', 'youngest_daughter'], 'Joy'),
            
            # ═══════════════════════════════════════════════════════════════
            # MYTHOLOGICAL (primes 653-719)
            # ═══════════════════════════════════════════════════════════════
            Symbol('ouroboros', 'Ouroboros', 653, SymbolCategory.MYTHOLOGICAL, '🐍',
                   ['eternity', 'cycle', 'unity'], 'Self-devouring serpent'),
            Symbol('world_tree', 'World Tree', 659, SymbolCategory.MYTHOLOGICAL, '🌳',
                   ['axis_mundi', 'connection', 'cosmos'], 'Axis of the world'),
            Symbol('grail', 'Holy Grail', 661, SymbolCategory.MYTHOLOGICAL, '🏆',
                   ['quest', 'enlightenment', 'sacred'], 'Sacred vessel'),
            Symbol('philosophers_stone', "Philosopher's Stone", 673, SymbolCategory.MYTHOLOGICAL, '💎',
                   ['transformation', 'perfection', 'alchemy'], 'Alchemical goal'),
            Symbol('caduceus', 'Caduceus', 677, SymbolCategory.MYTHOLOGICAL, '⚕',
                   ['healing', 'balance', 'mercury'], 'Staff of Hermes'),
            Symbol('ankh', 'Ankh', 683, SymbolCategory.MYTHOLOGICAL, '☥',
                   ['life', 'immortality', 'egyptian'], 'Key of life'),
            Symbol('mandala', 'Mandala', 691, SymbolCategory.MYTHOLOGICAL, '☸',
                   ['wholeness', 'center', 'cosmic'], 'Sacred circle'),
            Symbol('yin_yang', 'Yin Yang', 701, SymbolCategory.MYTHOLOGICAL, '☯',
                   ['duality', 'balance', 'taoist'], 'Unity of opposites'),
            Symbol('eye_providence', 'Eye of Providence', 709, SymbolCategory.MYTHOLOGICAL, '👁',
                   ['divine', 'watchful', 'all_seeing'], 'Divine eye'),
            Symbol('triskele', 'Triskele', 719, SymbolCategory.MYTHOLOGICAL, '☘',
                   ['cycles', 'celtic', 'trinity'], 'Triple spiral'),
        ]
        
        # Add all symbols to database
        for symbol in symbols:
            self.add_symbol(symbol)
    
    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the database"""
        self._symbols[symbol.id] = symbol
        self._by_prime[symbol.prime] = symbol
        self._by_name[symbol.name.lower()] = symbol
        self._by_category[symbol.category].append(symbol)
        
        for tag in symbol.cultural_tags:
            tag_lower = tag.lower()
            if tag_lower not in self._by_tag:
                self._by_tag[tag_lower] = []
            self._by_tag[tag_lower].append(symbol)
    
    def get_symbol(self, id_or_name: str) -> Optional[Symbol]:
        """Get symbol by ID or name"""
        # Try ID first
        if id_or_name in self._symbols:
            return self._symbols[id_or_name]
        # Try name
        return self._by_name.get(id_or_name.lower())
    
    def get_by_id(self, symbol_id: str) -> Optional[Symbol]:
        """Get symbol by ID"""
        return self._symbols.get(symbol_id)
    
    def get_symbol_by_prime(self, prime: int) -> Optional[Symbol]:
        """Get symbol by its associated prime"""
        return self._by_prime.get(prime)
    
    def get_symbols_by_category(self, category: SymbolCategory) -> List[Symbol]:
        """Get all symbols in a category"""
        return self._by_category.get(category, [])
    
    def get_symbols_by_tag(self, tag: str) -> List[Symbol]:
        """Get symbols with a specific cultural tag"""
        return self._by_tag.get(tag.lower(), [])
    
    def search(self, query: str, max_results: int = 10) -> List[Symbol]:
        """Search symbols by name or tags"""
        query = query.lower()
        results = []
        
        # Search in names
        for symbol in self._symbols.values():
            if query in symbol.name.lower() or query in symbol.id.lower():
                results.append(symbol)
                if len(results) >= max_results:
                    return results
        
        # Search in tags
        for symbol in self._symbols.values():
            if symbol not in results:
                for tag in symbol.cultural_tags:
                    if query in tag.lower():
                        results.append(symbol)
                        break
            if len(results) >= max_results:
                return results
        
        return results
    
    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols"""
        return list(self._symbols.values())
    
    def __len__(self) -> int:
        return len(self._symbols)
    
    def __contains__(self, symbol_id: str) -> bool:
        return symbol_id in self._symbols


# Singleton instance
symbol_database = SymbolDatabase()


class SemanticInference:
    """
    Semantic inference engine for symbol relationships.
    
    Uses prime relationships and cultural tags to infer
    connections between symbols.
    """
    
    def __init__(self, symbol_db: Optional[SymbolDatabase] = None):
        self.symbol_db = symbol_db or symbol_database
        self._phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    
    def infer_symbol(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Infer symbol properties from name or ID.
        
        Returns:
            Dict with symbol, method, confidence
        """
        symbol = self.symbol_db.get_symbol(name_or_id)
        if symbol:
            return {
                'symbol': symbol,
                'method': 'exact_match',
                'confidence': 1.0
            }
        
        # Try partial matching
        matches = self.symbol_db.search(name_or_id, max_results=5)
        if matches:
            return {
                'symbol': matches[0],
                'method': 'partial_match',
                'confidence': 0.7
            }
        
        return None
    
    def infer_with_resonance(self, text: str, options: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Infer symbols from text with resonance weighting.
        
        Args:
            text: Input text to analyze
            options: Optional configuration (maxCandidates, useAttention)
            
        Returns:
            List of {symbol, confidence, attentionWeight} dicts
        """
        options = options or {}
        max_candidates = options.get('maxCandidates', 10)
        
        # Tokenize and find matches
        tokens = text.lower().replace('.', ' ').replace(',', ' ').split()
        results = []
        seen = set()
        
        for token in tokens:
            if len(token) < 3:
                continue
            
            symbol = self.symbol_db.get_symbol(token)
            if symbol and symbol.id not in seen:
                seen.add(symbol.id)
                results.append({
                    'symbol': symbol,
                    'confidence': 1.0,
                    'attentionWeight': 1.0,
                    'source': token
                })
        
        # Sort by prime (lower primes = more fundamental)
        results.sort(key=lambda r: r['symbol'].prime)
        
        return results[:max_candidates]
    
    def infer_from_primes(self, primes: List[int], options: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Infer symbols from a list of primes.
        
        Args:
            primes: List of prime numbers
            options: Optional configuration
            
        Returns:
            List of inferences
        """
        options = options or {}
        max_depth = options.get('maxDepth', 2)
        
        results = []
        
        for prime in primes:
            # Direct match
            symbol = self.symbol_db.get_symbol_by_prime(prime)
            if symbol:
                results.append({
                    'type': 'direct',
                    'prime': prime,
                    'resultSymbol': symbol,
                    'confidence': 1.0
                })
            else:
                # Find nearest prime with symbol
                all_primes = sorted(self.symbol_db._by_prime.keys())
                if all_primes:
                    # Find closest
                    closest = min(all_primes, key=lambda p: abs(p - prime))
                    closest_symbol = self.symbol_db.get_symbol_by_prime(closest)
                    if closest_symbol:
                        distance = abs(closest - prime) / max(closest, prime)
                        results.append({
                            'type': 'nearest',
                            'prime': prime,
                            'nearestPrime': closest,
                            'resultSymbol': closest_symbol,
                            'confidence': max(0, 1 - distance)
                        })
        
        return results


class ResonanceCalculator:
    """
    Calculate resonance between primes using PHI-harmonic relationships.
    """
    
    def __init__(self):
        self._phi = (1 + math.sqrt(5)) / 2
    
    def calculate_resonance(self, p1: int, p2: int) -> float:
        """
        Calculate resonance between two primes.
        
        Uses the ratio's proximity to PHI and PHI powers.
        
        Args:
            p1: First prime
            p2: Second prime
            
        Returns:
            Resonance score 0-1
        """
        if p1 == p2:
            return 1.0
        
        ratio = max(p1, p2) / min(p1, p2)
        
        # Check proximity to PHI powers
        best_distance = abs(ratio - self._phi)
        
        for n in range(2, 8):
            phi_pow = self._phi ** n
            dist = abs(ratio - phi_pow)
            if dist < best_distance:
                best_distance = dist
        
        # Convert distance to resonance (exponential decay)
        resonance = math.exp(-best_distance * 0.5)
        
        return resonance
    
    def find_resonant_primes(self, prime: int, candidates: List[int], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find primes most resonant with the given prime.
        
        Args:
            prime: Target prime
            candidates: List of candidate primes
            top_n: Number of results to return
            
        Returns:
            List of {prime, resonance} dicts
        """
        scored = [
            {'prime': p, 'resonance': self.calculate_resonance(prime, p)}
            for p in candidates if p != prime
        ]
        
        scored.sort(key=lambda x: x['resonance'], reverse=True)
        return scored[:top_n]


class CompoundSymbol:
    """
    A compound symbol created from multiple base symbols.
    
    Represents complex concepts through symbol composition.
    """
    
    def __init__(
        self,
        id: str,
        components: List[Symbol],
        meaning: str,
        cultural_tags: Optional[List[str]] = None
    ):
        self.id = id
        self.components = components
        self.meaning = meaning
        self.cultural_tags = cultural_tags or []
        
        # Calculate combined prime (product of component primes, modulo a large prime)
        self.combined_prime = 1
        for s in components:
            self.combined_prime = (self.combined_prime * s.prime) % 10000019
        
        # Combined unicode
        self.unicode = ''.join(s.unicode for s in components[:3])
        
        self.created_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'components': [s.to_dict() for s in self.components],
            'meaning': self.meaning,
            'cultural_tags': self.cultural_tags,
            'combined_prime': self.combined_prime,
            'unicode': self.unicode
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Alias for to_dict for JS compatibility"""
        return self.to_dict()


class SymbolSequence:
    """
    A sequence of symbols representing temporal progression.
    """
    
    def __init__(self, symbols: Optional[List[Symbol]] = None):
        self.symbols = symbols or []
        self.timestamps: List[float] = []
    
    def append(self, symbol: Symbol, timestamp: Optional[float] = None) -> None:
        """Add a symbol to the sequence"""
        self.symbols.append(symbol)
        self.timestamps.append(timestamp or 0)
    
    def __len__(self) -> int:
        return len(self.symbols)
    
    def __iter__(self):
        return iter(self.symbols)


class CompoundBuilder:
    """
    Builder for compound symbols.
    """
    
    def __init__(self, symbol_db: Optional[SymbolDatabase] = None):
        self.symbol_db = symbol_db or symbol_database
        self.resonance_calc = ResonanceCalculator()
    
    def create_compound(
        self,
        compound_id: str,
        symbol_ids: List[str],
        meaning: str,
        cultural_tags: Optional[List[str]] = None
    ) -> CompoundSymbol:
        """
        Create a compound from symbol IDs.
        
        Args:
            compound_id: Unique ID for the compound
            symbol_ids: List of component symbol IDs
            meaning: Description of the compound meaning
            cultural_tags: Optional tags
            
        Returns:
            CompoundSymbol instance
        """
        components = []
        for sid in symbol_ids:
            symbol = self.symbol_db.get_symbol(sid)
            if symbol:
                components.append(symbol)
        
        if len(components) < 2:
            raise ValueError("Compound requires at least 2 symbols")
        
        return CompoundSymbol(compound_id, components, meaning, cultural_tags)
    
    def create_compound_from_symbols(
        self,
        compound_id: str,
        symbols: List[Symbol],
        meaning: str,
        cultural_tags: Optional[List[str]] = None
    ) -> CompoundSymbol:
        """
        Create a compound from Symbol instances.
        """
        if len(symbols) < 2:
            raise ValueError("Compound requires at least 2 symbols")
        
        return CompoundSymbol(compound_id, symbols, meaning, cultural_tags)
    
    def calculate_compound_resonance(self, compound: CompoundSymbol) -> float:
        """
        Calculate the internal resonance of a compound.
        
        Higher resonance = more harmonious combination.
        """
        if len(compound.components) < 2:
            return 0.0
        
        total_resonance = 0.0
        pairs = 0
        
        for i, s1 in enumerate(compound.components):
            for s2 in compound.components[i + 1:]:
                total_resonance += self.resonance_calc.calculate_resonance(s1.prime, s2.prime)
                pairs += 1
        
        return total_resonance / pairs if pairs > 0 else 0.0


# Singleton builder instance
compound_builder = CompoundBuilder()


class EntityExtractor:
    """
    Extract entities from text for symbol mapping.
    """
    
    def __init__(self):
        self.symbol_db = symbol_database
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of {entity, type, position} dicts
        """
        tokens = text.lower().split()
        entities = []
        
        for i, token in enumerate(tokens):
            # Clean token
            clean = ''.join(c for c in token if c.isalnum())
            if len(clean) < 2:
                continue
            
            symbol = self.symbol_db.get_symbol(clean)
            if symbol:
                entities.append({
                    'entity': clean,
                    'type': symbol.category.name,
                    'position': i,
                    'symbol': symbol
                })
        
        return entities