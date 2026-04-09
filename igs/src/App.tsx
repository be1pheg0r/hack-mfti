import { useMemo, useState } from 'react'
import './App.css'
import { LESSON_STEPS } from './data/lessons'
import { QuestionStepView } from './components/QuestionStepView'
import { StoryStepView } from './components/StoryStepView'

function App() {
  const [stepIndex, setStepIndex] = useState(0)
  const [selectedChoiceId, setSelectedChoiceId] = useState<string | null>(null)
  const [temporaryReaction, setTemporaryReaction] = useState<string | null>(null)

  const currentStep = LESSON_STEPS[stepIndex]
  const progressPercent = ((stepIndex) / LESSON_STEPS.length) * 100

  const selectedChoice = useMemo(() => {
    if (!currentStep || currentStep.kind !== 'question' || !selectedChoiceId) {
      return null
    }
    return currentStep.choices.find((choice) => choice.id === selectedChoiceId) ?? null
  }, [currentStep, selectedChoiceId])

  const isContinueDisabled =
    currentStep.kind === 'question' && !selectedChoice?.isCorrect

  const goToNextStep = () => {
    if (stepIndex >= LESSON_STEPS.length - 1) {
      setStepIndex(0)
      setSelectedChoiceId(null)
      setTemporaryReaction(null)
      return
    }
    setStepIndex((prev) => prev + 1)
    setSelectedChoiceId(null)
    setTemporaryReaction(null)
  }

  const handleContinue = () => {
    if (currentStep.kind === 'question' && !selectedChoice?.isCorrect) {
      return
    }
    goToNextStep()
  }

  const handleBack = () => {
    if (stepIndex > 0) {
      setStepIndex((prev) => prev - 1)
      setSelectedChoiceId(null)
      setTemporaryReaction(null)
    }
  }

  const handleSelectChoice = (choiceId: string) => {
    if (currentStep.kind !== 'question') {
      return
    }
    setSelectedChoiceId(choiceId)
    const choice = currentStep.choices.find((item) => item.id === choiceId)
    setTemporaryReaction(choice?.reaction ?? null)
  }

  return (
    <main className="lesson-page">
      <header className="lesson-header">
        <div className="progress-container">
          <button
            type="button"
            className="back-button"
            onClick={handleBack}
            disabled={stepIndex === 0}
            aria-label="Предыдущий шаг"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 5L7 10L12 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <div className="progress-track" aria-label="Прогресс урока">
            <div
              className="progress-fill"
              style={{ width: `${progressPercent}%` }}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progressPercent)}
            ></div>
          </div>
        </div>
      </header>

      <section className={`lesson-layout ${currentStep.kind === 'story' ? 'story-layout' : ''}`}>
        {currentStep.kind === 'question' ? (
          <QuestionStepView
            questionText={temporaryReaction ?? currentStep.question}
            mascotImage={selectedChoice?.image ?? currentStep.image}
            choices={currentStep.choices}
            selectedChoiceId={selectedChoiceId}
            onSelectChoice={handleSelectChoice}
            showQuestionButton={Boolean(temporaryReaction)}
            onShowQuestion={() => setTemporaryReaction(null)}
          />
        ) : (
          <StoryStepView step={currentStep} />
        )}
      </section>

      <footer className="lesson-footer">
        <button
          type="button"
          className="continue-button"
          onClick={handleContinue}
          disabled={isContinueDisabled}
        >
          Продолжить
        </button>
      </footer>
    </main>
  )
}

export default App
