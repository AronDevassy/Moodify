# BGM MASTER 🎵🧠

> **Let your mood pick the music.**

**BGM MASTER** is an offline MP3 music player that lets users organize their music into mood-based playlists and use **Moodify AI** to automatically choose music based on their facial expressions.

Built by **CtrlFreaks** for the **Useless Hackathon**.

---

Download the BGM_MASTER Demo.mp4 file to see the video demo

## 🎯 Basic Details

### Team Name

**CtrlFreaks**

### Team Members

- **Faheem Ahamed M S** — Muthoot Institute of Technology and Science  
  `25cs221@mgits.ac.in`

- **Aron Devassy** — Muthoot Institute of Technology and Science  
  `25cs021@mgits.ac.in`

### Hackathon

**Useless Hackathon**

---

# 💡 Project Description

**BGM MASTER** is a completely offline music player designed to make listening to music more personal.

The application works as a normal MP3 player, allowing users to add music files and folders, organize songs into playlists, and play them locally without requiring an internet connection.

The fun part comes with **Moodify**, our AI-powered mood feature.

When Moodify is enabled, the application uses the device camera to analyze the user's facial expressions and determine their current mood. BGM MASTER then selects suitable music from the corresponding mood playlist.

So instead of asking:

> **"What song should I play?"**

BGM MASTER asks:

> **"What mood are you in?"**

---

# 🤔 The Problem

Having hundreds of songs doesn't necessarily make choosing a song easier.

Users often spend more time deciding what to play than actually listening to music.

BGM MASTER solves this extremely serious problem by allowing the user's mood to influence the music selection.

---

# 💡 The Solution

BGM MASTER combines:

- Offline music playback
- Mood-based playlists
- Custom playlists
- Multiple user profiles
- Facial-expression analysis
- AI-powered mood detection
- Automatic song selection

The basic idea is:

```text
Face Expression
      ↓
Mood Detection
      ↓
Mood Playlist
      ↓
Song Selection
      ↓
Music Playback
```

---

# 🚀 Features

## 🎵 Offline MP3 Player

BGM MASTER works with music stored directly on the user's device.

Users can:

- Add individual music files
- Add entire music folders
- Browse their music library
- Play and pause songs
- Skip tracks
- Control volume
- Navigate playlists
- Use the application without internet access

---

## 😊 Mood-Based Playlists

Songs can be assigned to different moods.

Default moods include:

- ❤️ Romantic
- 😊 Happy
- 😔 Sad
- 🖤 Lonely
- 🌙 Chill
- ⚡ Excited

When a song is assigned a mood, it automatically becomes part of that mood's playlist.

Example:

```text
Happy
├── Song A
├── Song B
└── Song C
```

Clicking a mood opens its playlist and allows the user to browse and play its songs.

---

## 🧠 Moodify AI

**Moodify** is the AI feature of BGM MASTER.

When the user turns Moodify ON:

```text
Camera
   ↓
Facial Expression Analysis
   ↓
Detected Mood
   ↓
Matching Mood Playlist
   ↓
Song Selection
   ↓
Music Playback
```

The AI continuously observes facial expressions while music is playing and uses the detected mood when preparing upcoming tracks.

---

## 👤 Multiple Profiles

BGM MASTER supports multiple local user profiles.

Users can:

- Create profiles
- Select profiles
- Register facial-expression data
- Manage profiles
- Delete profiles

New profile registration includes an expression-capture process so the application can build the necessary facial-expression profile.

---

## 📁 Add Music & Folders

Users can either:

**Add Music**

or

**Add Folder**

Adding a folder allows multiple local music files to be imported into the library at once.

---

## 🎧 Custom Playlists

In addition to mood playlists, users can create their own playlists.

Examples:

- Favorites
- Workout
- Study
- Travel
- Late Night

A song can belong to multiple custom playlists without losing its mood assignment.

Custom playlists can also be pinned to the Home screen.

---

## 🎲 Random Mood-Based Selection

Moodify does not simply select the first song in a mood playlist.

It selects a song from the available songs in the detected mood while avoiding unnecessary immediate repetition when alternative tracks are available.

Example:

```text
Happy Playlist

Song A
Song B
Song C
Song D
```

If Song A is currently playing, Moodify can select another available song instead of immediately repeating Song A.

---

## 🔄 Automatic Queue

While a song is playing, Moodify continues analyzing expressions in the background.

Before the current song finishes:

```text
Current Song
      ↓
Current Mood
      ↓
Select Next Song
      ↓
Add to Queue
      ↓
Crossfade
      ↓
Next Song
```

This allows the listening experience to continue smoothly without repeatedly interrupting playback.

---

## 🎚️ Crossfade

BGM MASTER supports smooth transitions between songs.

Instead of abruptly stopping one song and starting another:

```text
Song A volume
100% → 0%

Song B volume
0% → 100%
```

This creates a smoother transition between tracks.

---

# 🖥️ User Interface

BGM MASTER uses **CustomTkinter** to provide a modern desktop interface.

The application includes:

- Sidebar navigation
- Profile selector
- Music library
- Mood playlists
- Custom playlists
- Moodify AI panel
- Camera preview
- Expression levels
- Now Playing bar
- Dark mode
- Light mode

The UI is designed around a combination of:

**Spotify-style music usability + Apple/macOS-inspired minimalism**

---

# 🧠 Moodify Registration

New profiles go through a facial-expression registration process.

The registration workflow is:

```text
Create Profile
      ↓
Enter Name
      ↓
Start Registration
      ↓
Camera
      ↓
Select Expression
      ↓
Capture Expression
      ↓
Minimum Samples Completed
      ↓
Optional Additional Samples
      ↓
Next Expression
      ↓
Complete Required Expressions
      ↓
Save Profile
```

Each required expression needs valid captured samples rather than simply moving through the registration process automatically.

---

# 📊 Expression Analysis

When Moodify is active, the right-side AI panel can display the detected expression levels.

Example:

```text
Happy        ███████████████░  82%
Sad          ██░░░░░░░░░░░░░  12%
Neutral      █░░░░░░░░░░░░░░   5%
Surprised    ░░░░░░░░░░░░░░░   1%
```

The values are generated from the existing AI-powered facial-expression analysis system.

---

# 🔐 Privacy

Privacy is an important part of BGM MASTER.

The application is designed around local processing.

- Camera processing happens locally.
- Facial-expression processing happens locally.
- Profile information is stored locally.
- Music remains on the user's device.
- Internet access is not required for normal operation.
- The camera is used only for the Moodify experience.

> **Your mood stays on your device.**

---

# 🛠️ Technical Details

## Languages

- Python

## UI Framework

- CustomTkinter (CTk)

## Computer Vision

- OpenCV
- MediaPipe

## AI / Machine Learning

- AI-trained facial-expression recognition system

## Audio

- Local MP3 / downloaded music files

## Storage

- Device-local storage

---

# 🏗️ System Architecture

BGM MASTER can be viewed as several connected modules.

```text
                         ┌───────────────────┐
                         │       USER        │
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │    BGM MASTER     │
                         │       APP         │
                         └─────────┬─────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               │                                       │
       ┌───────▼────────┐                    ┌─────────▼─────────┐
       │ Music Player   │                    │    Moodify AI     │
       └───────┬────────┘                    └─────────┬─────────┘
               │                                       │
               │                                ┌──────▼──────┐
               │                                │   Camera    │
               │                                └──────┬──────┘
               │                                       │
               │                                ┌──────▼──────┐
               │                                │ Expression  │
               │                                │  Analysis   │
               │                                └──────┬──────┘
               │                                       │
               │                                ┌──────▼──────┐
               │                                │ Mood Detect │
               │                                └──────┬──────┘
               │                                       │
               └────────────────┬──────────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Playlist  │
                         │   Selection │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │    Audio    │
                         │  Playback   │
                         └─────────────┘
```

---

# 🔄 Moodify Workflow

```text
MOODIFY ON
     ↓
Camera starts
     ↓
Facial expressions analyzed
     ↓
Mood determined
     ↓
Matching mood playlist selected
     ↓
Random suitable song selected
     ↓
Song starts
     ↓
Expressions continue being analyzed
     ↓
Next mood-based song prepared
     ↓
Added to queue
     ↓
Crossfade
     ↓
Next song
     ↓
Repeat
```

---

# 📦 Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY
cd BGM-MASTER
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

Start the application using the project's main Python entry file:

```bash
python main.py
```

If the main file has a different name, replace `main.py` with the appropriate entry point.

---

# 🎮 How To Use

## Normal Mode

```text
Open BGM MASTER
       ↓
Add Music / Add Folder
       ↓
Music Library
       ↓
Select a Song
       ↓
Play
```

## Moodify Mode

```text
Open BGM MASTER
       ↓
Turn ON Moodify AI
       ↓
Camera starts
       ↓
Facial expression analysis
       ↓
Mood detected
       ↓
Matching playlist selected
       ↓
Song selected
       ↓
Music plays
```

---

# 📸 Screenshots

Recommended structure:

### Home
<img width="960" height="600" alt="Screenshot 2026-09-04 101621" src="https://github.com/user-attachments/assets/1857b7e5-ecab-486b-93bf-de8c5ed22972" />


### Music Library

<img width="960" height="600" alt="Screenshot 2026-09-04 101630" src="https://github.com/user-attachments/assets/4a8acc86-eb94-483a-b1e6-943154e030ab" />


### Moodify AI

<img width="960" height="600" alt="Screenshot 2026-09-04 101649" src="https://github.com/user-attachments/assets/e3833e66-e509-4359-ba4f-eec74259b044" />


### Mood Playlists

<img width="960" height="600" alt="Screenshot 2026-09-04 101710" src="https://github.com/user-attachments/assets/a8870903-122a-431f-8b4a-801317de80cd" />


##DEMO VIDEO





# 🌱 Future Scope

Possible future improvements include:

- More advanced mood recommendations
- Improved facial-expression classification
- Smarter personalized playlists
- More music formats
- Better playlist customization
- Mobile application
- Voice-based controls
- More advanced AI music personalization
- Integration with online music services

---

# 🎯 Why BGM MASTER?

Because technology does not always need to solve a serious problem.

Sometimes it can simply make something ordinary a little more interesting.

**Music already matches our mood.**

We just made the mood choose the music.

---

# 👨‍💻 Team CtrlFreaks

### Faheem Ahamed M S

Muthoot Institute of Technology and Science  
`25cs221@mgits.ac.in`

### Aron Devassy

Muthoot Institute of Technology and Science  
`25cs021@mgits.ac.in`

---

# 🚀 Useless Hackathon

Built with curiosity, code, AI, and a completely unnecessary question:

> **"Can your face choose what song you should listen to?"**

### BGM MASTER 🎵

**Let your mood pick the music.**
