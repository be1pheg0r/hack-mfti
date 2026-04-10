import './Modal.css'
import ezhikImage from '../assets/ezhik_grid_2x5_clean.png'
import MASCOT_FRAMES from '../types/mascot'

export type ModalScenario = 'intro' | 'error' | 'success'

type ModalProps = {
  isOpen: boolean
  scenario: ModalScenario
  title: string
  paragraphs: string[]
  imageSrc: string
  imageAlt: string
  actionLabel: string
  onAction: () => void
}

export function Modal({
  isOpen,
  scenario,
  title,
  paragraphs,
  imageSrc,
  imageAlt,
  actionLabel,
  onAction,
}: ModalProps) {
  if (!isOpen) {
    return null
  }

  const neutralFrame = MASCOT_FRAMES.neutral
  const x = (neutralFrame.col / 4) * 100
  const y = neutralFrame.row * 100
  const rowOffset = neutralFrame.row === 1 ? -2 : 0

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={
        scenario === 'success'
          ? 'Окно завершения урока'
          : scenario === 'intro'
            ? 'Вступительное окно урока'
            : 'Окно с предупреждением об ошибках'
      }
    >
      <div className={`modal-card ${scenario === 'success' ? 'modal-card-success' : ''}`}>
        <p className="modal-message">{title}</p>
        {paragraphs.map((paragraph, index) => (
          <p key={index} className="modal-paragraph">
            {paragraph}
          </p>
        ))}
        <div className="modal-media-row">
          {scenario === 'intro' ? (
            <div className="modal-mascot-card" aria-hidden="true">
              <div
                className="modal-mascot-sprite"
                style={{
                  backgroundImage: `url(${ezhikImage})`,
                  backgroundPosition: `calc(${x}% + 0px) calc(${y}% + ${rowOffset}px)`,
                }}
              ></div>
            </div>
          ) : (
            <img
              className="modal-media"
              src={imageSrc}
              alt={imageAlt}
            />
          )}
        </div>

        <button type="button" className="modal-action-button" onClick={onAction}>
          {actionLabel}
        </button>
      </div>
    </div>
  )
}