import type { StoryStep } from '../types/lesson'
import heroImage from '../assets/hero.png'
import './StoryStepView.css'

type StoryStepViewProps = {
  step: StoryStep
}

export function StoryStepView({ step }: StoryStepViewProps) {
  return (
    <div className="story-center-wrap">
      <section className="story-panel" aria-label="Сценарий истории">
        
        <div className='story-image'>
          <div className='visual-label'>{step.visualLabel}</div>
          <img src={heroImage} width="160" height="160" alt="Изображение истории" />
        </div>

        <div className="story-content">
          <p className="story-text">{step.text}</p>
        </div>

      </section>
    </div>
  )
}
