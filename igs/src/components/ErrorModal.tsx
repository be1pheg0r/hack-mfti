import './ErrorModal.css'
export type ErrorModalScenario = 'error' | 'success'

type ErrorModalProps = {
  isOpen: boolean
  scenario: ErrorModalScenario
  title: string
  paragraphs: string[]
  imageSrc: string
  imageAlt: string
  actionLabel: string
  onAction: () => void
}

export function ErrorModal({
  isOpen,
  scenario,
  title,
  paragraphs,
  imageSrc,
  imageAlt,
  actionLabel,
  onAction,
}: ErrorModalProps) {
  if (!isOpen) {
    return null
  }

  return (
    <div
      className="error-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={scenario === 'success' ? 'Окно завершения урока' : 'Окно с предупреждением об ошибках'}
    >
      <div className={`error-modal-card ${scenario === 'success' ? 'error-modal-card-success' : ''}`}>
        <p className="error-modal-message">{title}</p>
        {paragraphs.map((paragraph, index) => (
          <p key={index} className="error-modal-paragraph">
            {paragraph}
          </p>
        ))}
        <div className="error-modal-media-row">
          <img
            className="error-modal-media"
            src={imageSrc}
            alt={imageAlt}
          />
        </div>

        <button type="button" className="error-modal-close-button" onClick={onAction}>
          {actionLabel}
        </button>
      </div>
    </div>
  )
}
