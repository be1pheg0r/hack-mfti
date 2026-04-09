import type { QuestionChoice } from '../types/lesson'
import MASCOT_FRAMES from '../types/mascot'
import ezhikImage from '../assets/ezhik_grid_2x5_clean.png'
import './QuestionStepView.css'

type QuestionStepViewProps = {
  questionText: string
  mascotImage: string
  choices: QuestionChoice[]
  selectedChoiceId: string | null
  onSelectChoice: (choiceId: string) => void
  showQuestionButton: boolean
  onShowQuestion: () => void
}

function getMascotFrame(image: string) {
  return MASCOT_FRAMES[image] ?? MASCOT_FRAMES.neutral
}

export function QuestionStepView({
  questionText,
  mascotImage,
  choices,
  selectedChoiceId,
  onSelectChoice,
  showQuestionButton,
  onShowQuestion,
}: QuestionStepViewProps) {
  const frame = getMascotFrame(mascotImage)
  const x = (frame.col / 4) * 100
  const y = frame.row * 100

  return (
    <>
      <div className="top-section">
        <div className="character-container">
          <div className="character-card" aria-hidden="true">
            <div
              className="character-sprite"
              style={{
                backgroundImage: `url(${ezhikImage})`,
                backgroundPosition: `${x}% ${y}%`,
              }}
            ></div>
          </div>
        </div>

        <div className="right-column">
          <h1 className="question-text">{questionText}</h1>
          {showQuestionButton && (
            <button type="button" className="show-question-button" onClick={onShowQuestion}>
              показать вопрос
            </button>
          )}
        </div>
      </div>

      <div className="answers-section">
        <section className="answers-panel" aria-label="Варианты ответа">
          {choices.map((choice) => {
            const isSelected = selectedChoiceId === choice.id
            return (
              <button
                key={choice.id}
                type="button"
                className={`choice-button ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectChoice(choice.id)}
              >
                {choice.text}
              </button>
            )
          })}
        </section>
      </div>
    </>
  )
}
