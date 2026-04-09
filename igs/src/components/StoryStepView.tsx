import type { StoryStep } from '../types/lesson'
import heroImage from '../assets/hero.png'
import './StoryStepView.css'

type StoryStepViewProps = {
  step: StoryStep
}

const STORY_MEDIA_SOURCES = import.meta.glob('../assets/*.{png,jpg,jpeg,gif,webp,mp4,webm,ogg}', {
  eager: true,
  import: 'default',
}) as Record<string, string>

function resolveStoryMediaSrc(mediaKey: string) {
  const normalizedKey = mediaKey.trim()
  if (!normalizedKey) {
    return undefined
  }

  const directMatch = STORY_MEDIA_SOURCES[normalizedKey]
  if (directMatch) {
    const extension = normalizedKey.split('.').pop()?.toLowerCase() ?? ''
    return {
      src: directMatch,
      isVideo: extension === 'mp4' || extension === 'webm' || extension === 'ogg',
    }
  }

  const entry = Object.entries(STORY_MEDIA_SOURCES).find(([filePath]) => {
    const fileName = filePath.split('/').pop() ?? ''
    const baseName = fileName.replace(/\.[^.]+$/, '')
    return normalizedKey === fileName || normalizedKey === baseName
  })

  if (!entry) {
    return undefined
  }

  const fileName = entry[0].split('/').pop() ?? ''
  const extension = fileName.split('.').pop()?.toLowerCase() ?? ''

  return {
    src: entry[1],
    isVideo: extension === 'mp4' || extension === 'webm' || extension === 'ogg',
  }
}

export function StoryStepView({ step }: StoryStepViewProps) {
  const storyMedia = resolveStoryMediaSrc(step.image)

  return (
    <div className="story-center-wrap">
      <section className="story-panel" aria-label="Сценарий истории">
        
        <div className='story-image'>
          <div className='visual-label'>{step.visualLabel}</div>
          {storyMedia ? (
            storyMedia.isVideo ? (
              <video
                src={storyMedia.src}
                className="story-media"
                autoPlay
                loop
                muted
                playsInline
                preload="metadata"
                aria-label="Видео истории"
              />
            ) : (
              <img src={storyMedia.src} className="story-media" alt="Изображение истории" />
            )
          ) : (
            <img src={heroImage} className="story-media" alt="Изображение истории" />
          )}
        </div>

        <div className="story-content">
          <p className="story-text">{step.text}</p>
        </div>

      </section>
    </div>
  )
}
