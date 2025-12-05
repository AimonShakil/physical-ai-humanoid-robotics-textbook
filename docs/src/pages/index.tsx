import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className={styles.heroBackground}>
        <div className={styles.heroGradient}></div>
        <div className={styles.heroParticles}>
          <div className={styles.particle}></div>
          <div className={styles.particle}></div>
          <div className={styles.particle}></div>
        </div>
      </div>
      <div className={clsx('container', styles.heroContainer)}>
        <div className={styles.heroContent}>
          <Heading as="h1" className={styles.heroTitle}>
            {siteConfig.title}
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
          <div className={styles.heroStats}>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>32+</div>
              <div className={styles.statLabel}>Chapters</div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>3</div>
              <div className={styles.statLabel}>Modules</div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>AI</div>
              <div className={styles.statLabel}>Powered</div>
            </div>
          </div>
          <div className={styles.heroButtons}>
            <Link
              className={clsx('button button--primary button--lg', styles.primaryButton)}
              to="/docs/intro">
              Start Learning 🚀
            </Link>
            <Link
              className={clsx('button button--secondary button--lg', styles.secondaryButton)}
              to="/docs/module1-ros2/intro">
              Explore ROS 2
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({title, description, icon}: {title: string; description: string; icon: string}) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.featureIcon}>{icon}</div>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
    </div>
  );
}

function FeaturesSection() {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            Why Choose This Textbook?
          </Heading>
          <p className={styles.sectionSubtitle}>
            Comprehensive learning path from basics to advanced Physical AI
          </p>
        </div>
        <div className={styles.featuresGrid}>
          <FeatureCard
            icon="🤖"
            title="ROS 2 Mastery"
            description="Learn Robot Operating System 2 from scratch with hands-on examples and best practices"
          />
          <FeatureCard
            icon="🦾"
            title="Humanoid Robotics"
            description="Master bipedal locomotion, inverse kinematics, and whole-body control systems"
          />
          <FeatureCard
            icon="🧠"
            title="Physical AI"
            description="Explore embodied intelligence, reinforcement learning, and LLMs for robotics"
          />
          <FeatureCard
            icon="💬"
            title="AI Chatbot Assistant"
            description="Get instant answers with our RAG-powered chatbot that knows the entire textbook"
          />
          <FeatureCard
            icon="📝"
            title="Interactive Learning"
            description="Select any text and ask questions instantly with our intelligent assistant"
          />
          <FeatureCard
            icon="🎨"
            title="Beautiful Design"
            description="Modern UI with light/dark mode, smooth animations, and premium aesthetics"
          />
        </div>
      </div>
    </section>
  );
}

function ModulesSection() {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            Learning Modules
          </Heading>
          <p className={styles.sectionSubtitle}>
            Structured curriculum designed for progressive learning
          </p>
        </div>
        <div className={styles.modulesGrid}>
          <div className={styles.moduleCard}>
            <div className={styles.moduleNumber}>01</div>
            <h3>ROS 2 Fundamentals</h3>
            <p>Core concepts, nodes, topics, services, and navigation</p>
            <Link to="/docs/module1-ros2/intro" className={styles.moduleLink}>
              Start Module →
            </Link>
          </div>
          <div className={styles.moduleCard}>
            <div className={styles.moduleNumber}>02</div>
            <h3>Humanoid Robotics</h3>
            <p>Anatomy, kinematics, balance, and bipedal walking</p>
            <Link to="/docs/module2-humanoid-robotics/intro" className={styles.moduleLink}>
              Start Module →
            </Link>
          </div>
          <div className={styles.moduleCard}>
            <div className={styles.moduleNumber}>03</div>
            <h3>Physical AI</h3>
            <p>Embodied intelligence, sensor fusion, and RL</p>
            <Link to="/docs/module3-physical-ai/intro" className={styles.moduleLink}>
              Start Module →
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaContent}>
          <Heading as="h2" className={styles.ctaTitle}>
            Ready to Master Physical AI?
          </Heading>
          <p className={styles.ctaSubtitle}>
            Join thousands of learners building the future of robotics
          </p>
          <Link
            className={clsx('button button--primary button--lg', styles.ctaButton)}
            to="/docs/intro">
            Start Your Journey 🚀
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Comprehensive textbook on ROS 2, humanoid robotics, and embodied AI">
      <HomepageHeader />
      <main>
        <FeaturesSection />
        <ModulesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
