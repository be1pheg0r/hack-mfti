import { useMemo, useState } from 'react'
import './App.css'

import blehGif from './assets/bleh.gif'
import blueFaceGif from './assets/blue_face.gif'
// import hamsterGif from './assets/hamster.gif'

import { ErrorModal, type ErrorModalScenario } from './components/ErrorModal'
import { LESSON_STEPS } from './data/lessons'
import { QuestionStepView } from './components/QuestionStepView'
import { StoryStepView } from './components/StoryStepView'

type ModalContent = {
  title: string
  paragraphs: string[]
  imageSrc: string
  imageAlt: string
  actionLabel: string
}

const MODAL_CONTENT: Record<ErrorModalScenario, ModalContent> = {
  error: {
    title: 'Мирного пути не будет',
    paragraphs: [
      'Ты много раз неправильно ответил на вопросы. Поэтому мы должны удалить твой компьютер. К сожалению, это единственный способ, чтобы ты смог продолжить обучение и стать настоящим гением.',
      'Ладно, это шутка. Просто сфокусируйся и попробуй снова. У тебя все получится!',
    ],
    imageSrc: blueFaceGif, // blueFaceGif hamsterGif
    imageAlt: 'Hamster motivator',
    actionLabel: 'Продолжить',
  },
  success: {
    title: 'Отличная работа! Ты прошел все шаги',
    paragraphs: ['Ты завершил урок. Можешь закрепить результат и начать сначала, чтобы чувствовать себя еще увереннее.'],
    imageSrc: blehGif, 
    imageAlt: 'Celebration gif',
    actionLabel: 'Начать сначала',
  },
}

function App() {
  const [stepIndex, setStepIndex] = useState(0)
  const [selectedChoiceId, setSelectedChoiceId] = useState<string | null>(null)
  const [temporaryReaction, setTemporaryReaction] = useState<string | null>(null)
  const [wrongAttemptsCount, setWrongAttemptsCount] = useState(0)
  const [modalScenario, setModalScenario] = useState<ErrorModalScenario | null>(null)

  const currentStep = LESSON_STEPS[stepIndex]
  const progressPercent = ((stepIndex) / (LESSON_STEPS.length - 1)) * 100
  const maxWrongAttempts = 7
  const errorCount = wrongAttemptsCount
  const errorPercent = maxWrongAttempts > 0 ? (errorCount / maxWrongAttempts) * 100 : 0
  const isLastStep = stepIndex === LESSON_STEPS.length - 1
  const activeModalContent = modalScenario ? MODAL_CONTENT[modalScenario] : null

  const selectedChoice = useMemo(() => {
    if (!currentStep || currentStep.kind !== 'question' || !selectedChoiceId) {
      return null
    }
    return currentStep.choices.find((choice) => choice.id === selectedChoiceId) ?? null
  }, [currentStep, selectedChoiceId])

  const isQuestionStep = currentStep.kind === 'question'
  const isReactionVisible = isQuestionStep && Boolean(temporaryReaction)
  const isCorrectChoiceChecked = isQuestionStep && isReactionVisible && Boolean(selectedChoice?.isCorrect)

  const primaryButtonLabel =
    isQuestionStep && !isCorrectChoiceChecked ? 'Ответить' : 'Продолжить'

  const primaryButtonClassName =
    isQuestionStep && isReactionVisible
      ? selectedChoice?.isCorrect
        ? 'continue-button continue-button-correct'
        : 'continue-button continue-button-wrong'
      : 'continue-button'

  const isPrimaryButtonDisabled = isQuestionStep ? !selectedChoice : false

  const goToNextStep = () => {
    if (isLastStep) {
      return
    }
    setStepIndex((prev) => prev + 1)
    setSelectedChoiceId(null)
    setTemporaryReaction(null)
  }

  const resetLesson = () => {
    setStepIndex(0)
    setSelectedChoiceId(null)
    setTemporaryReaction(null)
    setWrongAttemptsCount(0)
    setModalScenario(null)
  }

  const handlePrimaryAction = () => {
    if (!isQuestionStep) {
      if (isLastStep) {
        setModalScenario('success')
        return
      }
      goToNextStep()
      return
    }

    if (!selectedChoice) {
      return
    }

    if (isCorrectChoiceChecked) {
      if (isLastStep) {
        setModalScenario('success')
        return
      }
      goToNextStep()
      return
    }

    if (!selectedChoice.isCorrect) {
      const nextWrongAttemptsCount = wrongAttemptsCount + 1
      if (maxWrongAttempts > 0 && nextWrongAttemptsCount >= maxWrongAttempts) {
        setWrongAttemptsCount(0)
        setModalScenario('error')
      } else {
        setWrongAttemptsCount(nextWrongAttemptsCount)
      }
    }

    setTemporaryReaction(selectedChoice.reaction)
  }

  const handleBack = () => {
    if (stepIndex > 0) {
      setStepIndex((prev) => prev - 1)
      setSelectedChoiceId(null)
      setTemporaryReaction(null)
    }
  }

  const handleSelectChoice = (choiceId: string) => {
    if (!isQuestionStep) {
      return
    }
    setSelectedChoiceId(choiceId)
    setTemporaryReaction(null)
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
            mascotImage={isReactionVisible && selectedChoice ? selectedChoice.image : currentStep.image}
            choices={currentStep.choices}
            selectedChoiceId={selectedChoiceId}
            onSelectChoice={handleSelectChoice}
            showQuestionButton={Boolean(temporaryReaction)}
            onShowQuestion={() => setTemporaryReaction(null)}
            errorPercent={errorPercent}
          />
        ) : (
          <StoryStepView step={currentStep} />
        )}
      </section>

      <footer className="lesson-footer">
        <button
          type="button"
          className={primaryButtonClassName}
          onClick={handlePrimaryAction}
          disabled={isPrimaryButtonDisabled}
        >
          {primaryButtonLabel}
        </button>
      </footer>

      {modalScenario && activeModalContent && (
        <ErrorModal
          isOpen={true}
          scenario={modalScenario}
          title={activeModalContent.title}
          paragraphs={activeModalContent.paragraphs}
          imageSrc={activeModalContent.imageSrc}
          imageAlt={activeModalContent.imageAlt}
          actionLabel={activeModalContent.actionLabel}
          onAction={() => {
            if (modalScenario === 'success') {
              resetLesson()
              return
            }
            setModalScenario(null)
          }}
        />
      )}
    </main>
  )
}

export default App
