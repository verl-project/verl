# UFC Fight Prediction Environment

This environment provides a framework for training and evaluating AI models on UFC fight prediction tasks, with a unique twist: instead of traditional analytical predictions, it generates entertaining fight commentary that can be directly used with Text-to-Speech (TTS) models like DIA. The environment includes two main components: a text-based predictor and an image-based predictor, both designed to create engaging, broadcast-style fight commentary.

## Environment Design

### Core Components

1. **UFC Server (ufc_server.py)**
   - Text-based fight prediction environment
   - Generates dynamic, entertaining fight commentary
   - Uses fighter statistics and historical data
   - Outputs TTS-ready commentary with dramatic flair
   - Implements a scoring system for model evaluation

2. **UFC Image Environment (ufc_image_env.py)**
   - Visual-based fight prediction environment
   - Creates commentary based on fighter appearances
   - Implements multimodal prediction capabilities
   - Generates broadcast-style commentary from visual analysis
   - Includes image processing and base64 encoding utilities

### Data Structure

- **fighter_stats.csv**: Contains detailed statistics for each fighter including:
  - Win/Loss records
  - Physical attributes (height, weight, reach)
  - Performance metrics (strikes per minute, takedown accuracy, etc.)

- **large_dataset.csv**: Historical fight data including:
  - Fighter matchups
  - Fight outcomes
  - Event information

- **fighter_images/**: Directory containing fighter profile images
  - Images are stored in JPG format
  - Filenames follow slug format (e.g., "john-smith.jpg")

## Motivation

This environment was designed to transform traditional fight prediction into an engaging entertainment experience:

1. **Entertainment-First Approach**
   - Generates dynamic, broadcast-style fight commentary
   - Creates TTS-ready output for voice synthesis
   - Incorporates dramatic elements and commentator personalities
   - Makes fight prediction more engaging and accessible

2. **Statistical Analysis with Style**
   - Wraps technical analysis in entertaining commentary
   - Uses fight statistics to inform dramatic storytelling
   - Maintains prediction accuracy while being entertaining
   - Creates a more engaging way to present fight analysis

3. **Visual Storytelling**
   - Transforms visual analysis into engaging commentary
   - Creates dramatic narratives from fighter appearances
   - Makes technical observations more accessible
   - Generates TTS-compatible descriptions of visual elements

4. **Multimodal Entertainment**
   - Combines statistical and visual data for rich commentary
   - Creates cohesive narratives from multiple data sources
   - Generates engaging stories that work well with TTS
   - Makes technical analysis more accessible and fun

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
   - Ensure fighter_stats.csv and large_dataset.csv are in the environment directory
   - Place fighter images in the fighter_images/ directory

3. Run the environment:
   - For text-based commentary: Use UFCEnv
   - For image-based commentary: Use UFCImageEnv

4. TTS Integration:
   - The generated commentary is formatted for direct use with TTS models
   - Includes dramatic pauses and emphasis markers
   - Contains natural speech patterns and commentator personalities
   - Ready for voice synthesis with models like DIA

## Example Runs

Here are some example runs demonstrating the environment in action:

- [Video Demo](https://youtu.be/C_hFe6TfQvU) - Watch the environment in action with real-time commentary generation
- [Text-based Prediction Run](https://wandb.ai/edtheman/Atropos-environments_ufc_env/runs/rq5wfxgh?nw=nwuseredtheman) - Shows the environment generating commentary based on fighter statistics and historical data
- [Image-based Prediction Run](https://wandb.ai/edtheman/Atropos-environments_ufc_env/runs/klw4m5of?nw=nwuseredtheman) - Demonstrates the environment creating commentary from visual analysis of fighter appearances

The key difference between these runs is their input modality:
- The text-based run focuses on statistical analysis and historical data to generate commentary
- The image-based run analyzes fighter appearances and visual characteristics to create engaging narratives

## Configuration

The environment can be configured through the following parameters:

- `fighter_stats_path`: Path to fighter statistics CSV
- `fight_data_path`: Path to fight dataset CSV
- `image_folder`: Path to fighter images directory
- `max_steps`: Number of steps per prediction
- `temperature`: Generation diversity parameter (affects commentary style)
- `top_p`: Nucleus sampling parameter (affects commentary creativity)

## Scoring System

The environment implements a scoring system that evaluates predictions based on:
- Accuracy of winner prediction
- Entertainment value of the commentary
- TTS compatibility and natural flow
- Integration of statistical/visual data in an engaging way
- Proper formatting for voice synthesis

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- New commentary styles and personalities
- Enhanced TTS compatibility features
- Additional dramatic elements
- Improved entertainment value
- Better integration with voice synthesis models
