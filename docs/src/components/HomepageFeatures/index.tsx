import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
  link: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: ROS 2',
    emoji: '📚',
    link: '/docs/module1-ros2/intro',
    description: (
      <>
        Master ROS 2, the industry standard for robot development. Learn nodes,
        topics, services, and build production-ready robot applications.
      </>
    ),
  },
  {
    title: 'Module 2: Humanoid Robotics',
    emoji: '🤖',
    link: '/docs/module2-humanoid-robotics/intro',
    description: (
      <>
        Explore humanoid robot design, bipedal walking, balance control, and
        whole-body motion planning for human-like robots.
      </>
    ),
  },
  {
    title: 'Module 3: Physical AI',
    emoji: '🧠',
    link: '/docs/module3-physical-ai/intro',
    description: (
      <>
        Integrate AI with robotics: computer vision, reinforcement learning,
        LLMs for robotics, and multi-modal embodied intelligence.
      </>
    ),
  },
];

function Feature({title, emoji, description, link}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <div style={{fontSize: '4rem', marginBottom: '1rem'}}>{emoji}</div>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <a href={link} className="button button--primary">
          Explore Module →
        </a>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
